import numpy as np
import math
from .faa_model_base import Faa_model_base

class faa_rot_model(Faa_model_base):
    def __init__(self, config):
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
        x_rot = y0 * np.sin(psi)
        y_rot = y0 * np.cos(psi)
        z_rot = z0
        
        # 合并坐标
        antenna_positions_rotated = np.column_stack((x_rot, y_rot, z_rot))
        
        
        return antenna_positions_rotated

    def directional_channel_gain(self, antenna_positions_rotated):
        """
        计算定向天线的信道增益
        
        参数:
            config: 配置对象，包含天线和信道参数
            antenna_positions_rotated: 旋转后的天线位置坐标
            
        返回:
            定向天线信道向量
        """
        # 计算旋转后的方位角
        new_phi = self.config.phi_paths - math.radians(self.config.psi)
        
        # 向量化计算所有路径的方向性增益
        A_dir = np.array([super()._calculate_GE(t, p, self.config.Kappa) 
                          for t, p in zip(self.config.theta_paths, new_phi)])
        
        # 预分配信道向量
        h_vector_dir = np.zeros(len(antenna_positions_rotated), dtype=complex)
        
        # 计算每个天线的信道增益
        for n, antenna_pos in enumerate(antenna_positions_rotated):
            # 向量化计算所有路径的贡献
            path_responses = np.array([self.config.beta_paths[i] * A_dir[i] * 
                                    super()._calculate_path_phase_diff(
                                          self.config.theta_paths[i], 
                                          self.config.phi_paths[i], 
                                          antenna_pos)
                                      for i in range(self.config.L)])
            
            # 求和并归一化
            h_vector_dir[n] = np.sqrt(1 / self.config.L) * np.sum(path_responses)
        
        print("定向天线信道\n", h_vector_dir)
        return h_vector_dir
    def omni_channel_gain(self, antenna_positions_rotated):
        A_onmi = np.ones(self.config.L, dtype=complex)
        # 预分配信道向量
        h_vector_onmi = np.zeros(len(antenna_positions_rotated), dtype=complex)
        
        # 计算每个天线的信道增益
        for n, antenna_pos in enumerate(antenna_positions_rotated):
            # 向量化计算所有路径的贡献
            path_responses = np.array([self.config.beta_paths[i] * A_onmi[i] * 
                                      super()._calculate_path_phase_diff(
                                          self.config.theta_paths[i], 
                                          self.config.phi_paths[i], 
                                          antenna_pos)
                                      for i in range(self.config.L)])
            
            # 求和并归一化
            h_vector_onmi[n] = np.sqrt(1 / self.config.L) * np.sum(path_responses)
        
        print("全向天线信道\n", h_vector_onmi)
        return h_vector_onmi
