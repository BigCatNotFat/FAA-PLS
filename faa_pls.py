import numpy as np
import math
from models.faa_rot_model import faa_rot_model
from config.config import Config

class Faa_pls:
    def __init__(self, config, model):
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
        # 随机生成L个俯仰角 (0到π)
        self.__theta_paths = config.theta_paths
        # 随机生成L个方位角 (-π到π)
        self.__phi_paths = config.phi_paths
        # 天线阵列旋转角度
        self.__psi = config.psi
        # 原始天线位置坐标（未旋转）
        self.__antenna_positions_original = []
        # 变换后的天线位置坐标
        self.__antenna_positions_transform = []
        # 初始化波束赋形向量，功率均匀分布在所有天线上
        self.__w = np.ones(self.__N_total) * np.sqrt(self.__P_max / self.__N_total)
        self.model = model
        self.channel_gain_directional_bob = []
        self.channel_gain_omni_bob = []
        self.channel_gain_directional_eve = []
        self.channel_gain_omni_eve = []
        # ## 生成响应系数sigma，g_0表示参考距离1 m处平均信道功率增益的期望值，d_k表示距离，loss_exp表示路径损耗因子
        # g_0 = 10 ** (-40 / 10)
        # d_k = 100
        # loss_exp = 2
        # sigma_real = (g_0 / (d_k ** loss_exp)) / L
        # print(sigma_real)
        # beta_paths = np.sqrt(sigma_real / 2) * (np.random.randn(L) + 1j * np.random.randn(L))
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
        
        print("【原始】天线坐标 (未旋转):")
        for i, (x0, y0, z0) in enumerate(self.__antenna_positions_original, start=1):
            print(f"  天线单元 {i}: (x, y, z) = ({x0:.3f}, {y0:.3f}, {z0:.3f})")
    def transform_antenna_positions(self, degree):

        #生成旋转后的天线坐标
        self.__antenna_positions_transform = self.model.transform_antenna_positions(self.__antenna_positions_original, degree)
        print("【旋转后】天线坐标 (绕 z 轴旋转 ψ):")
        for i, (x0, y0, z0) in enumerate(self.__antenna_positions_transform, start=1):
            print(f"  天线单元 {i}: (x, y, z) = ({x0:.3f}, {y0:.3f}, {z0:.3f})")
    def generate_channel_Bob(self):
        self.channel_gain_directional_bob  = self.model.directional_channel_gain(self.__antenna_positions_transform, self.config.beta_paths_bob)
        self.channel_gain_omni_bob = self.model.omni_channel_gain(self.__antenna_positions_transform, self.config.beta_paths_bob)
        print("Bob定向天线信道\n", self.channel_gain_directional_bob)
        print("Bob全向天线信道\n", self.channel_gain_omni_bob)
    def generate_channel_Eve(self):
        self.channel_gain_directional_eve = self.model.directional_channel_gain(self.__antenna_positions_transform, self.config.beta_paths_eve)
        self.channel_gain_omni_eve = self.model.omni_channel_gain(self.__antenna_positions_transform, self.config.beta_paths_eve)
        print("Eve定向天线信道\n", self.channel_gain_directional_eve)
        print("Eve全向天线信道\n", self.channel_gain_omni_eve)
    def solve(self):
        pass


def main():
    #初始化配置
    config = Config()
    rot_model = faa_rot_model(config)
    faa_rot = Faa_pls(config, rot_model)
    faa_rot.generate_antenna_original_positions()
    faa_rot.transform_antenna_positions(30)#传入psi参数
    faa_rot.generate_channel_Bob()
    faa_rot.generate_channel_Eve()


if __name__ == "__main__":
    main()




