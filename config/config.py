import numpy as np


class Config:
    def __init__(self):
        # 设置随机数种子以确保结果可重现
        self.seed = 43
        np.random.seed(self.seed)
        
        # 基本参数设置
        self.fre = 10e9       # 频率 (Hz)
        self.lambda_ = 0.03   # 波长 (m)
        self.d = self.lambda_ / 2  # 天线间距，设为半波长
        
        # 天线阵列配置
        self.N_v = 3          # 垂直方向天线数量
        self.N_h = 3          # 水平方向天线数量
        self.N = self.N_v * self.N_h
        # 计算总天线数
        self.N_total = self.N_v * self.N_h
        self.L = 10             #每个天线的信道数量
        self.Kappa = 4          #锐度因子
        self.P_max = 10**(0/10)  #最大信道功率0dbm      #最大信道功率0dbm (1 mW)          
        self.psi = 90          #旋转角度
        self.theta_paths_bob = np.random.uniform(0, np.pi, size=self.L)
        self.phi_paths_bob = np.random.uniform(-np.pi, np.pi, size=self.L)
        self.theta_paths_eve = np.random.uniform(0, np.pi, size=self.L)
        self.phi_paths_eve = np.random.uniform(-np.pi, np.pi, size=self.L)
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
        
        # 日志配置
        self.log_enabled = True  # 是否启用日志
        self.log_level = "INFO"  # 日志级别: DEBUG, INFO, WARNING, ERROR
        self.log_to_file = True  # 是否输出到文件
        self.log_file_path = "logs/app.log"  # 日志文件路径
        self.log_console_output = True  # 是否在控制台输出日志
        #假设bob的eve的噪声功率相同
        self.sigma_sq = 10 ** (-92 / 10)


        
        # Adam optimizer parameters for psi
        self.adam_lr_psi = 0.5          # Learning rate for psi (applied to degrees)
        self.adam_beta1_psi = 0.9
        self.adam_beta2_psi = 0.999
        self.adam_epsilon_psi = 1e-8
        self.psi_opt_iterations = 1000    # Number of Adam iterations for psi in each alternating step
        
        # Alternating optimization
        self.num_alternating_iterations = 100 # Number of outer loops (w -> psi -> w -> psi ...)
        