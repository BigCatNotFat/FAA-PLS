import numpy as np
from abc import ABC, abstractmethod
class Faa_model_base(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def transform_antenna_positions(self):
        pass

    @abstractmethod
    def directional_channel_gain(self):
        # 默认实现，子类可以覆盖此方法
        pass

    @abstractmethod
    def omni_channel_gain(self):
        pass

    def _calculate_GE(self, theta, phi, kappa=4):
        """
        计算方向性增益元素（公式8）
        
        参数:
            theta: 俯仰角（弧度）
            phi: 方位角（弧度）
            kappa: 锐度因子，默认为4
            
        返回:
            方向性增益值
        """
        # 只在有效角度范围内计算增益
        if -np.pi/2 <= phi <= np.pi/2 and 0 <= theta <= np.pi:
            G = 2 * (1 + kappa)
            return np.sqrt(G * (np.sin(theta) ** kappa) * (np.cos(phi) ** kappa))
        else:
            return 0
        
    def _calculate_path_phase_diff(self, theta, phi, antenna_positions):
        """
        计算某个天线某一条路径的相位差，对应论文里面的公式3
        
        参数:
            theta: 俯仰角（弧度），范围[0, π]
            phi: 方位角（弧度），范围[-π, π]
            antenna_positions: 天线位置坐标，形式为[x, y, z]
            
        原理:
            根据入射波的方向和天线位置，计算相对于参考点的相位差
            k = 2π/λ 是波数
            相位差 = exp(-j·k·(投影到入射方向的距离))
            
        返回:
            g_n: 复数形式的相位差
        """
        # 计算波数 k (rad/m)
        k = 2 * np.pi / self.config.lambda_
        g_n = np.exp(-1j * k * (
                antenna_positions[0] * np.sin(theta) * np.cos(phi) +
                antenna_positions[1] * np.sin(theta) * np.sin(phi) +
                antenna_positions[2] * np.cos(theta)))
        return g_n