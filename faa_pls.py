import numpy as np
import math
from models.faa_rot_model import faa_rot_model
from config.config import Config
import logs.logger as logger
from optimizers.opt import Optimizer


class Faa_pls:
    def __init__(self, config, model, opt):
        # 存储配置对象
        self.config = config
        # 垂直方向天线数量
        self.__N_v = config.N_v
        # 水平方向天线数量
        self.__N_h = config.N_h
        # 波长
        self.__lambda_ = config.lambda_
        # 天线间距
        self.__d = config.d
        # 每个天线的信道数量
        self.__L = config.L
        # 最大信道功率
        self.__P_max = config.P_max
        # 锐度因子
        self.__Kappa = config.Kappa
        # 总天线数
        self.__N_total = config.N_total
        # 天线阵列旋转角度
        self.__psi = config.psi
        # 原始天线位置坐标（未旋转）
        self.__antenna_positions_original = []
        # 变换后的天线位置坐标
        self.__antenna_positions_transform = []
        self.__sigma_sq = config.sigma_sq
        # 初始化波束赋形向量，功率均匀分布在所有天线上
        self.__w = np.ones(self.__N_total) * np.sqrt(self.__P_max / self.__N_total)
        self.model = model
        self.channel_gain_directional_bob = []
        self.channel_gain_omni_bob = []
        self.channel_gain_directional_eve = []
        self.channel_gain_omni_eve = []
        self.channel_drivate_bob = []
        self.channel_drivate_eve = []
        self.__psi_degrees = 0
        self.__gradient_F_psi = 0
        self.optimizer = opt


    def generate_antenna_original_positions(self):
        #生成天线坐标
        #生成原始天线坐标，在未旋转前，x=0，z轴为中间位置
        # 使用向量化操作生成天线坐标
        nh_indices = np.arange(1, self.__N_h + 1)
        nv_indices = np.arange(1, self.__N_v + 1)
        
        # 计算y和z坐标
        y_coords = ((2 * nh_indices - self.__N_h - 1) / 2) * self.__d
        z_coords = ((2 * nv_indices - self.__N_v - 1) / 2) * self.__d
        
        # 使用网格生成所有坐标组合
        y_grid, z_grid = np.meshgrid(y_coords, z_coords)
        x_grid = np.zeros_like(y_grid)
        
        # 重塑并组合坐标
        self.__antenna_positions_original = np.column_stack((
            x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
        ))
        
        logger.info("【原始】天线坐标 (未旋转):",self.__antenna_positions_original)

    def transform_antenna_positions(self):

        #生成旋转后的天线坐标
        self.__antenna_positions_transform = self.model.transform_antenna_positions(self.__antenna_positions_original, self.__psi_degrees)
        logger.info(f"【旋转后】天线坐标 (绕 z 轴旋转 {self.__psi_degrees} 度):",self.__antenna_positions_transform)

    def generate_channel_bob(self):
        self.channel_gain_directional_bob, self.channel_drivate_bob  = self.model.get_channel_and_derivative(self.__antenna_positions_transform, self.config.beta_paths_bob, self.config.theta_paths_bob, self.config.phi_paths_bob, self.__psi_degrees)
        self.channel_gain_omni_bob = self.model.omni_channel_gain(self.__antenna_positions_transform, self.config.beta_paths_bob, self.config.theta_paths_bob, self.config.phi_paths_bob)
        
        
        logger.info("bob定向天线信道", self.channel_gain_directional_bob)
        # logger.info("bob全向天线信道", self.channel_gain_omni_bob)
    def generate_channel_eve(self):
        self.channel_gain_directional_eve , self.channel_drivate_eve= self.model.get_channel_and_derivative(self.__antenna_positions_transform, self.config.beta_paths_eve, self.config.theta_paths_eve, self.config.phi_paths_eve, self.__psi_degrees)
        self.channel_gain_omni_eve = self.model.omni_channel_gain(self.__antenna_positions_transform, self.config.beta_paths_eve, self.config.theta_paths_eve, self.config.phi_paths_eve)
        
        
        logger.info("eve定向天线信道", self.channel_gain_directional_eve)
        # logger.info("eve全向天线信道", self.channel_gain_omni_eve)

    def get_gradient_F_psi(self):
        self.__gradient_F_psi =   self.model.psi_derivative(self.channel_gain_directional_bob, self.channel_gain_directional_eve, self.channel_drivate_bob, self.channel_drivate_eve, self.__w)
        logger.info("梯度更新", self.__gradient_F_psi)
        return self.__gradient_F_psi

    def opt_w(self):
        self.__w = self.optimizer.optimize_beamformer_fixed_psi(self.channel_gain_directional_bob, self.channel_gain_directional_eve, self.__P_max, self.__sigma_sq)
        logger.info("波束赋形向量更新", self.__w)

    def opt_psi(self):
        for t in range(1, self.config.psi_opt_iterations + 1):
            self.get_gradient_F_psi()
            self.__psi_degrees = self.optimizer.optimize_psi_fixed_beam(self.__psi_degrees, self.__w, self.__gradient_F_psi, t)
        


    def calculate_objective_and_secrecy_rate(self):
        # Ensure channels are up-to-date with current self.__psi_degrees
        self.transform_antenna_positions() # uses self.__psi_degrees
        self.generate_channel_bob()      # uses self.__psi_degrees & self.__antenna_positions_transform
        self.generate_channel_eve()      # uses self.__psi_degrees & self.__antenna_positions_transform

        # Ensure w is shaped as (N,1) for matrix product if h is (N,1)
        w_col = self.__w.reshape(-1,1)

        S_B = np.abs(np.conjugate(self.channel_gain_directional_bob.T) @ w_col)**2
        S_E = np.abs(np.conjugate(self.channel_gain_directional_eve.T) @ w_col)**2
        
        # S_B and S_E are (1,1) arrays, extract scalar
        S_B_scalar = S_B[0,0]
        S_E_scalar = S_E[0,0]

        objective_F = (self.__sigma_sq + S_B_scalar) / (self.__sigma_sq + S_E_scalar) \
                      if (self.__sigma_sq + S_E_scalar) != 0 else float('inf')
        
        SNR_B = S_B_scalar / self.__sigma_sq if self.__sigma_sq > 0 else float('inf')
        SNR_E = S_E_scalar / self.__sigma_sq if self.__sigma_sq > 0 else float('inf')

        # Secrecy rate (bits per channel use)
        # C_s = max(0, log2(1+SNR_B) - log2(1+SNR_E))
        # If SNR_B or SNR_E can be very small or zero, handle log2(1+x) carefully
        term_B = np.log2(1 + SNR_B) if (1 + SNR_B) > 0 else -float('inf') 
        term_E = np.log2(1 + SNR_E) if (1 + SNR_E) > 0 else -float('inf')
        
        secrecy_rate = term_B - term_E
        # secrecy_rate = max(0, secrecy_rate) # Typically, secrecy rate is non-negative

        return objective_F, secrecy_rate, S_B_scalar, S_E_scalar
    def solve(self):
        self.generate_antenna_original_positions()
        # self.transform_antenna_positions()#传入psi参数
        # self.generate_channel_bob()
        # self.generate_channel_eve()
        # self.opt_w()
        # self.get_gradient_F_psi()
        # self.opt_psi()

        for alt_iter in range(1, self.config.num_alternating_iterations + 1):
            logger.info(f"--- Alternating Optimization Iteration: {alt_iter}/{self.config.num_alternating_iterations} ---")
            
            # 1. Update channels based on current self.__psi_degrees
            self.transform_antenna_positions() 
            self.generate_channel_bob()      
            self.generate_channel_eve()      
            
            # 2. Optimize w for fixed psi (updates self.__w)
            self.opt_w()

            # 3. Optimize psi for fixed w (updates self.__psi_degrees)
            self.opt_psi()

            # 4. Log current state
            #    The call to opt_psi already updated self.__psi_degrees.
            #    Channels need to be consistent with this final psi and the optimized w.
            #    The calculate_objective_and_secrecy_rate method handles recalculating channels
            #    based on the latest self.__psi_degrees.
            obj_F, sr, s_b, s_e = self.calculate_objective_and_secrecy_rate()
            
            logger.info(f"End of Alt. Iter {alt_iter}: psi = {self.__psi_degrees:.2f} deg, Objective F = {obj_F:.4f}")
            logger.info(f"                           S_B = {s_b:.3e}, S_E = {s_e:.3e}, Secrecy Rate ~ {sr:.4f} bits/s/Hz")
            norm_w_sq = np.linalg.norm(self.__w)**2
            logger.info(f"                           Beamformer power: {norm_w_sq:.4f} (P_max: {self.__P_max:.4f})")
            if not np.isclose(norm_w_sq, self.__P_max):
                 logger.warning(f"                           Beamformer power {norm_w_sq:.4e} not equal to P_max {self.__P_max:.4e}")         

        



def main():
    #初始化配置
    mconfig = Config()
    #设置日志打印工具
    logger.init_logger(mconfig)
    #设置优化器
    optimizer = Optimizer(mconfig)
    #设置模型
    rot_model = faa_rot_model(mconfig)
    #传入参数
    faa_rot = Faa_pls(mconfig, rot_model, optimizer)
    #解优化问题
    faa_rot.solve()

if __name__ == "__main__":
    main()




