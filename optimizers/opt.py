import numpy as np
import scipy.linalg
import logs.logger as logger

class Optimizer:
    def __init__(self, config) -> None:
        self.config = config

    def optimize_beamformer_fixed_psi(self, h_B, h_E, P_max, sigma_sq): 
        """
        优化固定阵列形状参数psi下的发射波束成形向量w。
        这对应于论文第III-A节中的子问题求解。

        参数:
            h_B (np.ndarray): Bob的信道向量(复数，形状(N,))。
                              假设为固定psi下的h_B(psi)。
            h_E (np.ndarray): Eve的信道向量(复数，形状(N,))。
                              假设为固定psi下的h_E(psi)。
            P_max (float): 最大发射功率。
            sigma_sq (float): 噪声方差(sigma^2)，假设Bob和Eve相同。

        返回:
            np.ndarray: 最优波束成形向量w_star(复数，形状(N,))。
        """
        N = h_B.shape[0] # 天线数量
        if h_E.shape[0] != N:
            raise ValueError()
            logger.info("h_B和h_E必须具有相同的维度(N_antennas)\n")

        if P_max <= 0:
            raise ValueError("P_max必须为正数。")
        if sigma_sq <= 0: # c需要sigma_sq > 0，否则如果h_E为零，B可能是奇异的
            logger.info("警告：sigma_sq理想情况下应为正数，以使c定义良好且B稳健正定。")
            # 如果sigma_sq为0，c为0。问题结构略有变化。
            # 广义瑞利商方法仍然适用，但B = h_E h_E^H。
            # 如果h_E也为0，B是奇异的。这是一个退化情况。
            # 对于实际场景，sigma_sq > 0。
            if sigma_sq == 0 and np.allclose(h_E, 0):
                 # 如果没有噪声且Eve的信道为零，任何不与h_B正交的波束成形器都会给Bob提供无限SNR_B
                 # 如果Eve的信道不为零，则SNR_E也为无限大。
                 # 如果Eve的信道为零且Bob的不为零，则SNR_E=0，保密率为C_B。通过MRT最大化C_B。
                 logger.info("退化情况：sigma_sq为0且h_E为0。使用Bob的MRT。")
                 norm_h_B = np.linalg.norm(h_B)
                 if norm_h_B > 1e-9:
                     return np.sqrt(P_max) * h_B / norm_h_B
                 else: # Bob的信道也为零
                     return np.sqrt(P_max / N) * np.ones(N, dtype=complex) # 任意选择

        c = sigma_sq / P_max

        h_B_col = h_B.reshape(-1, 1)
        h_E_col = h_E.reshape(-1, 1)
        
        I = np.eye(N, dtype=complex)
        
        A_matrix = h_B_col @ h_B_col.conj().T + c * I
        B_matrix = h_E_col @ h_E_col.conj().T + c * I

        try:
            eigenvalues, eigenvectors = scipy.linalg.eig(A_matrix, B_matrix)
        except scipy.linalg.LinAlgError as e:
            logger.info(f"广义特征值分解过程中的线性代数错误：{e}")
            # 如果B是奇异的(例如，c=0且h_E秩亏损或为零)的备选方案
            # 如果B在数值上是奇异的，可以尝试添加一个小的正则化项
            # B_matrix_reg = B_matrix + 1e-12 * I 
            # eigenvalues, eigenvectors = scipy.linalg.eig(A_matrix, B_matrix_reg)
            # 或者，如果h_E确实为零且c为零，这意味着Eve没有接收到信号。
            # 那么问题就只是最大化Bob的SNR。
            if np.allclose(B_matrix, 0): # 实际上意味着h_E为零且c为零。
                logger.info("B_matrix接近零。这意味着Eve的信道为空且噪声相对于P_max可忽略不计。")
                logger.info("通过MRT最大化Bob的SNR。")
                norm_h_B = np.linalg.norm(h_B)
                if norm_h_B > 1e-9:
                     return np.sqrt(P_max) * h_B / norm_h_B
                else: # Bob的信道也为零
                     return np.sqrt(P_max / N) * np.ones(N, dtype=complex) # 任意选择
            raise e

        real_eigenvalues = np.real(eigenvalues)
        idx_dominant_eigenvalue = np.argmax(real_eigenvalues)
        v_max_dominant = eigenvectors[:, idx_dominant_eigenvalue]

        norm_v_max = np.linalg.norm(v_max_dominant)
        if norm_v_max < 1e-9:
            # 如果A不为零，这种情况理想情况下不应该发生。
            # 如果A为零(h_B为零且c为零)，则任何向量都是特征值为0的特征向量。
            # 这意味着Bob没有接收到信号。保密率将<= 0。
            # 返回一个具有正确功率的任意向量。
            logger.info("警告：主特征向量的范数接近零。返回任意波束成形器。")
            return np.sqrt(P_max / N) * np.ones(N, dtype=complex) 
        
        v_max_normalized = v_max_dominant / norm_v_max
        w_star = np.sqrt(P_max) * v_max_normalized
        return w_star
    

    def optimize_psi_fixed_beam(self, 
                                    current_psi_degrees_initial: float, 
                                    w_fixed: np.ndarray, 
                                    gradient_F_psi: np.ndarray,
                                    t:int
                                   # func(psi_degrees, w_fixed) -> gradient_radians
                                ):
            """
            Optimizes psi (in degrees) for a fixed beamforming vector w_fixed using Adam.
            The objective is to MAXIMIZE F(psi) = (sigma^2 + S_B(psi)) / (sigma^2 + S_E(psi)).
            Adam is a minimizer, so we minimize -F(psi).
            The gradient_calculator_func must return dF/d(psi_radians).

            Args:
                current_psi_degrees_initial (float): Initial guess for psi in degrees.
                w_fixed (np.ndarray): The fixed beamforming vector, shape (N,).
                gradient_calculator_func (callable): A function that takes (psi_degrees, w_fixed_array)
                                                    and returns the gradient dF/d(psi_radians).

            Returns:
                float: Optimized psi in degrees.
            """
            psi_degrees = float(current_psi_degrees_initial) # Ensure it's a float
            
            m_psi = 0.0  # First moment
            v_psi = 0.0  # Second moment (uncentered variance)
            
            # Adam hyperparameters from config
            lr = self.config.adam_lr_psi
            beta1 = self.config.adam_beta1_psi
            beta2 = self.config.adam_beta2_psi
            epsilon = self.config.adam_epsilon_psi
            num_iterations = self.config.psi_opt_iterations

                # 1. Calculate gradient dF/d(psi_radians) using the callback
                # The callback handles antenna transforms, channel calculations for the current_psi_degrees
            if np.isnan(gradient_F_psi):
                    logger.warning(f"Optimizer: Gradient is NaN. Stopping psi optimization early.")
                    
                
                # 2. Convert gradient for Adam step:
                # We want to MAXIMIZE F(psi). Adam MINIMIZES Loss(psi).
                # So, Loss(psi) = -F(psi).
                # d(Loss)/d(psi_degrees) = d(-F)/d(psi_radians) * d(psi_radians)/d(psi_degrees)
                #                        = - (dF/d(psi_radians)) * (pi / 180.0)
            gradient_for_adam_update = -gradient_F_psi * (np.pi / 180.0)

                # 3. Adam update steps
            m_psi = beta1 * m_psi + (1 - beta1) * gradient_for_adam_update
            v_psi = beta2 * v_psi + (1 - beta2) * (gradient_for_adam_update ** 2)
                
                # Bias correction
            m_hat_psi = m_psi / (1 - beta1 ** t)
            v_hat_psi = v_psi / (1 - beta2 ** t)
                
                # Parameter update (Adam subtracts the scaled gradient)
            update_step_degrees = lr * m_hat_psi / (np.sqrt(v_hat_psi) + epsilon)
            psi_degrees -= update_step_degrees

                # Normalize psi to [0, 360) range, or a more physically relevant range like [0, 180)
                # For UPA, rotation from 0 to 180 degrees often covers all unique orientations.
                # Let's use [0, 180) for now. Consider if problem implies wider range.
                # Given G_E has cos(phi_eff)^kappa, if kappa is even, (psi) and (psi+180) might be related.
                # For now, [0, 360) is safe.
            psi_degrees = psi_degrees % 360.0
                # If specific range e.g. [0, 90] is desired, clipping or modulo to that range would be here.
                # self.__psi_degrees = np.clip(self.__psi_degrees, 0, 90)
            logger.info(f"Optimizer: Finished Psi optimization. Final psi: {psi_degrees:.2f} deg.")
            return psi_degrees
