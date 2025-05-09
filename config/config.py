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
        self.beta_paths = (np.random.randn(self.L) + 1j * np.random.randn(self.L)) / np.sqrt(2)
