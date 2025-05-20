import numpy as np
import math
from .faa_model_base import Faa_model_base
from logs import logger

class faa_rot_model(Faa_model_base):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def transform_antenna_positions(self, antenna_positions_original, psi):
        """
        计算天线阵列绕z轴旋转后的坐标
        
        参数:
            antenna_positions_original: 原始天线坐标数组
            psi: 旋转角度（弧度）
        
        返回:
            旋转后的天线坐标数组
        """
        psi = math.radians(psi)
        # 使用向量化操作代替循环
        positions = np.array(antenna_positions_original)
        
        # 提取坐标分量
        x0, y0, z0 = positions[:, 0], positions[:, 1], positions[:, 2]
        
        # 计算旋转后坐标
        x_rot = x0*np.cos(psi) - y0 * np.sin(psi)
        y_rot = x0*np.sin(psi) + y0 * np.cos(psi)
        z_rot = z0
        
        # 合并坐标
        antenna_positions_rotated = np.column_stack((x_rot, y_rot, z_rot))
        return antenna_positions_rotated

    def directional_channel_gain(self, antenna_positions_rotated, beta_paths, theta_paths, phi_paths, psi):
        """
        计算定向天线的信道增益
        
        参数:
            config: 配置对象，包含天线和信道参数
            antenna_positions_rotated: 旋转后的天线位置坐标
            
        返回:
            定向天线信道向量
        """
        # 计算旋转后的方位角
        new_phi = phi_paths - math.radians(psi)
        
        # 向量化计算所有路径的方向性增益
        A_dir = np.array([self._calculate_GE(t, p, self.config.Kappa) 
                          for t, p in zip(theta_paths, new_phi)])
        
        # 预分配信道向量
        h_vector_dir = np.zeros(len(antenna_positions_rotated), dtype=complex)
        
        # 计算每个天线的信道增益
        for n, antenna_pos in enumerate(antenna_positions_rotated):
            # 向量化计算所有路径的贡献
            path_responses = np.array([beta_paths[i] * A_dir[i] * 
                                    self._calculate_path_phase_diff(
                                          theta_paths[i], 
                                          phi_paths[i], 
                                          antenna_pos)
                                      for i in range(self.config.L)])
            
            # 求和并归一化
            h_vector_dir[n] = np.sqrt(1 / self.config.L) * np.sum(path_responses)
        return h_vector_dir
    

    def omni_channel_gain(self, antenna_positions_rotated, beta_paths, theta_paths, phi_paths):
        A_onmi = np.ones(self.config.L, dtype=complex)
        # 预分配信道向量
        h_vector_onmi = np.zeros(len(antenna_positions_rotated), dtype=complex)
        
        # 计算每个天线的信道增益
        for n in range(len(antenna_positions_rotated)):
            antenna_pos = antenna_positions_rotated[n]
            # 使用ndarray向量化计算所有路径的贡献
            phase_diffs = np.array([self._calculate_path_phase_diff(
                                    theta_paths[i], 
                                    phi_paths[i], 
                                    antenna_pos) for i in range(self.config.L)])
            
            path_responses = beta_paths * A_onmi * phase_diffs
            
            # 求和并归一化
            h_vector_onmi[n] = np.sqrt(1 / self.config.L) * np.sum(path_responses)
        
        return h_vector_onmi

    def psi_derivative(self,h_B_psi,h_E_psi,dh_B_dpsi,dh_E_dpsi,w_fixed):
        w_fixed = np.array(w_fixed).reshape(-1, 1)
        # --- Calculate S_B(psi), S_E(psi) ---
        # S_B(psi) = |h_B(psi)^H w|^2
        hB_H_w = np.conjugate(h_B_psi.T) @ w_fixed # This is a scalar (1x1 matrix)
        S_B_psi = np.abs(hB_H_w[0,0])**2

        hE_H_w = np.conjugate(h_E_psi.T) @ w_fixed # This is a scalar (1x1 matrix)
        S_E_psi = np.abs(hE_H_w[0,0])**2

        # --- Calculate dS_B(psi)/dpsi, dS_E(psi)/dpsi ---
        # dS_B(psi)/dpsi = 2 * Re{ ( (dh_B(psi)/dpsi)^H w ) * ( w^H h_B(psi) ) }
        # Note: w^H h_B(psi) = (h_B(psi)^H w)^*
        
        dhB_dpsi_H_w = np.conjugate(dh_B_dpsi.T) @ w_fixed # scalar
        w_H_hB_psi = np.conjugate(w_fixed.T) @ h_B_psi     # scalar
        
        dS_B_dpsi = 2 * np.real(dhB_dpsi_H_w[0,0] * w_H_hB_psi[0,0])

        dhE_dpsi_H_w = np.conjugate(dh_E_dpsi.T) @ w_fixed # scalar
        w_H_hE_psi = np.conjugate(w_fixed.T) @ h_E_psi     # scalar
        
        dS_E_dpsi = 2 * np.real(dhE_dpsi_H_w[0,0] * w_H_hE_psi[0,0])

        # --- Final Gradient dF(psi)/dpsi ---
        # dF/dpsi = ( (dS_B/dpsi)*(sigma^2 + S_E) - (sigma^2 + S_B)*(dS_E/dpsi) ) / (sigma^2 + S_E)^2
        
        numerator = dS_B_dpsi * (self.config.sigma_sq + S_E_psi) - (self.config.sigma_sq + S_B_psi) * dS_E_dpsi
        denominator = (self.config.sigma_sq + S_E_psi)**2
        if denominator == 0:
            # Avoid division by zero; this case should ideally not happen if sigma_sq > 0
            # Or if S_E is always non-negative.
            # If it happens, it might indicate an issue or a need for regularization.
            # For now, return a large number or handle as an error.
            return np.nan # Or handle appropriately

        gradient_F_psi = numerator / denominator
        
        return gradient_F_psi

    def get_channel_and_derivative(self, antenna_positions_rotated, beta_paths, theta_paths, phi_paths, psi_degrees):
        h_psi = np.zeros((self.config.N, 1), dtype=np.complex128)
        dh_dpsi = np.zeros((self.config.N, 1), dtype=np.complex128)
        k_wave = 2 * np.pi / self.config.lambda_
        G_gain = 2 * (1 + self.config.Kappa)
        kappa = self.config.Kappa
        psi = math.radians(psi_degrees)  # 仅计算一次，供后续使用

        for l_idx in range(self.config.L):
            alpha_l = beta_paths[l_idx]
            phi_l = phi_paths[l_idx]
            theta_l = theta_paths[l_idx]

            # Wave vector k_l
            k_l_vec = k_wave * np.array([
                np.sin(theta_l) * np.cos(phi_l),
                np.sin(theta_l) * np.sin(phi_l),
                np.cos(theta_l)
            ])

            # ---------------- 元件方向增益及其关于 psi 的导数 ----------------
            phi_eff_l = phi_l - psi  # 有效方位角
            cos_phi_eff_l = np.cos(phi_eff_l)
            sin_phi_eff_l = np.sin(phi_eff_l)

            # 调用父类公共方法计算增益
            g_l_n_psi_val = self._calculate_GE(theta_l, phi_eff_l, kappa)

            # 导数: dg/dpsi = g * (kappa/2) * tan(phi_eff)
            if g_l_n_psi_val != 0.0 and cos_phi_eff_l != 0.0:
                dg_l_n_dpsi_val = g_l_n_psi_val * (kappa / 2.0) * (sin_phi_eff_l / cos_phi_eff_l)
            else:
                dg_l_n_dpsi_val = 0.0

            # ---------------- 逐天线计算 ----------------
            for n_v_idx in range(self.config.N_v):
                for n_h_idx in range(self.config.N_h):
                    n_idx = n_v_idx * self.config.N_h + n_h_idx

                    C_nh = (2 * (n_h_idx + 1) - self.config.N_h - 1) / 2
                    # 直接使用外部传入的旋转后坐标，而无需重新计算
                    r_n_psi = antenna_positions_rotated[n_idx]

                    # r_n_psi 对 psi 的导数
                    dr_n_dpsi = np.array([
                        -C_nh * self.config.d * np.cos(psi),
                        -C_nh * self.config.d * np.sin(psi),
                        0.0
                    ])

                    # 利用父类方法计算相位差 a_l_n_psi
                    a_l_n_psi = self._calculate_path_phase_diff(theta_l, phi_l, r_n_psi)

                    # a_l_n_psi 关于 psi 的导数
                    k_dot_dr_dpsi = np.dot(k_l_vec, dr_n_dpsi)
                    da_l_n_dpsi = a_l_n_psi * (-1j * k_dot_dr_dpsi)

                    # 组合得到 c_l_n_psi 及其导数
                    c_l_n_psi = g_l_n_psi_val * a_l_n_psi
                    dc_l_n_dpsi = dg_l_n_dpsi_val * a_l_n_psi + g_l_n_psi_val * da_l_n_dpsi

                    h_psi[n_idx] += alpha_l * c_l_n_psi
                    dh_dpsi[n_idx] += alpha_l * dc_l_n_dpsi
        
        norm_factor = np.sqrt(1.0 / self.config.L) if self.config.L > 0 else 1.0
        h_psi *= norm_factor
        dh_dpsi *= norm_factor
        return h_psi, dh_dpsi
    

