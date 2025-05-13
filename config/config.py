import numpy as np

class Config:
    def __init__(self):
        # 基本参数设置
        self.fre = 10e9       # 频率 (Hz)
        self.lambda_ = 0.03   # 波长 (m)
        self.d = self.lambda_ / 2  # 天线间距，设为半波长
        
        # 天线阵列配置
        self.N_v = 2          # 垂直方向天线数量
        self.N_h = 3          # 水平方向天线数量
        
        # 计算总天线数
        self.N_total = self.N_v * self.N_h
        self.L = 10             #每个天线的信道数量
        self.Kappa = 4          #锐度因子
        self.P_max = 10**(0/10)  #最大信道功率0dbm         
        self.psi = 90          #旋转角度
        self.theta_paths = np.random.uniform(0, np.pi, size=self.L)
        self.phi_paths = np.random.uniform(-np.pi, np.pi, size=self.L)
        # self.beta_paths = (np.random.randn(self.L) + 1j * np.random.randn(self.L)) / np.sqrt(2)
        # 生成响应系数sigma，g_0表示参考距离1 m处平均信道功率增益的期望值，d_k表示距离，loss_exp表示路径损耗因子
        # Bob信道参数
        self.g_0_bob = 10 ** (-40 / 10)
        self.d_k_bob = 50
        self.loss_exp_bob = 2
        self.sigma_real_bob = (self.g_0_bob / (self.d_k_bob ** self.loss_exp_bob)) / self.L
        self.beta_paths_bob = np.sqrt(self.sigma_real_bob / 2) * (np.random.randn(self.L) + 1j * np.random.randn(self.L))
        # Eve信道参数
        self.g_0_eve = 10 ** (-40 / 10)
        self.d_k_eve = 50
        self.loss_exp_eve = 2
        self.sigma_real_eve = (self.g_0_eve / (self.d_k_eve ** self.loss_exp_eve)) / self.L
        self.beta_paths_eve = np.sqrt(self.sigma_real_eve / 2) * (np.random.randn(self.L) + 1j * np.random.randn(self.L))