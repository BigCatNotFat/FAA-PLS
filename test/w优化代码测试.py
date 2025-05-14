import numpy as np
import scipy.linalg # 用于广义特征值问题

# --- (之前的 optimize_beamformer_fixed_psi 函数代码保持不变) ---
def optimize_beamformer_fixed_psi(h_B, h_E, P_max, sigma_sq):
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
        raise ValueError("h_B和h_E必须具有相同的维度(N_antennas)")

    if P_max <= 0:
        raise ValueError("P_max必须为正数。")
    if sigma_sq <= 0: # c需要sigma_sq > 0，否则如果h_E为零，B可能是奇异的
        print("警告：sigma_sq理想情况下应为正数，以使c定义良好且B稳健正定。")
        # 如果sigma_sq为0，c为0。问题结构略有变化。
        # 广义瑞利商方法仍然适用，但B = h_E h_E^H。
        # 如果h_E也为0，B是奇异的。这是一个退化情况。
        # 对于实际场景，sigma_sq > 0。
        if sigma_sq == 0 and np.allclose(h_E, 0):
             # 如果没有噪声且Eve的信道为零，任何不与h_B正交的波束成形器都会给Bob提供无限SNR_B
             # 如果Eve的信道不为零，则SNR_E也为无限大。
             # 如果Eve的信道为零且Bob的不为零，则SNR_E=0，保密率为C_B。通过MRT最大化C_B。
             print("退化情况：sigma_sq为0且h_E为0。使用Bob的MRT。")
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
        print(f"广义特征值分解过程中的线性代数错误：{e}")
        # 如果B是奇异的(例如，c=0且h_E秩亏损或为零)的备选方案
        # 如果B在数值上是奇异的，可以尝试添加一个小的正则化项
        # B_matrix_reg = B_matrix + 1e-12 * I 
        # eigenvalues, eigenvectors = scipy.linalg.eig(A_matrix, B_matrix_reg)
        # 或者，如果h_E确实为零且c为零，这意味着Eve没有接收到信号。
        # 那么问题就只是最大化Bob的SNR。
        if np.allclose(B_matrix, 0): # 实际上意味着h_E为零且c为零。
            print("B_matrix接近零。这意味着Eve的信道为空且噪声相对于P_max可忽略不计。")
            print("通过MRT最大化Bob的SNR。")
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
        print("警告：主特征向量的范数接近零。返回任意波束成形器。")
        return np.sqrt(P_max / N) * np.ones(N, dtype=complex) 
    
    v_max_normalized = v_max_dominant / norm_v_max
    w_star = np.sqrt(P_max) * v_max_normalized
    return w_star

def calculate_secrecy_rate(w, h_B, h_E, sigma_sq):
    """计算给定波束成形器w的保密率。"""
    if sigma_sq == 0: # 处理无噪声情况以避免除以零
        gamma_B_val = np.abs(h_B.conj() @ w)**2 / 1e-12 # 使用一个极小数而不是零
        gamma_E_val = np.abs(h_E.conj() @ w)**2 / 1e-12
        if np.abs(h_B.conj() @ w)**2 == 0: gamma_B_val = 0
        if np.abs(h_E.conj() @ w)**2 == 0: gamma_E_val = 0

        # 如果gamma_E为0，1+gamma_E = 1。如果gamma_B非常大，速率就很大。
        # 如果gamma_E也非常大，比值可能会很棘手。
        # 对于log2((1+gamma_B)/(1+gamma_E))，如果sigma_sq -> 0，则c -> 0
        # 如果两者都不为零，它变成log2(|h_B^H w|^2 / |h_E^H w|^2)
        # 即2 * log2(|h_B^H w| / |h_E^H w|)
        # 如果h_E^H w为零，这可能会导致问题。
        # 为了数值稳定性和一致性，我们坚持使用(1+gamma)公式。
        # 如果sigma_sq真的为零，容量的定义可能需要谨慎处理。
        # 对于模拟，我们假设sigma_sq > 0。
        # 如果用户提供sigma_sq=0，这部分需要稳健处理或警告。
        # 目前，我们假设在实践中sigma_sq将> 0。
        # 如果不是，问题就不那么关于(1+SNR)，而更多关于SNR比率。

    if sigma_sq <= 1e-12: # 实际上零噪声
        # 如果Eve的信道增益为零，log(1/0)是个问题
        val_bob = np.abs(h_B.conj() @ w)**2
        val_eve = np.abs(h_E.conj() @ w)**2
        if val_eve < 1e-12: # Eve几乎没有接收到信号
            if val_bob < 1e-12: # Bob也没有接收到信号
                return 0.0
            else: # Bob接收到信号，Eve没有 -> 对于小sigma_sq，实际上是无限正速率
                  # 为避免无穷大，我们考虑速率 = log2(1 + val_bob/sigma_sq)
                  # 如果原始值为0，让我们使用一个非常小的sigma_sq进行计算
                  effective_sigma_sq = 1e-12
                  gamma_B_val = val_bob / effective_sigma_sq
                  return np.log2(1 + gamma_B_val) # gamma_E为0
        # 如果val_eve不为零，标准公式有效
        effective_sigma_sq = sigma_sq if sigma_sq > 1e-12 else 1e-12
        gamma_B_val = val_bob / effective_sigma_sq
        gamma_E_val = val_eve / effective_sigma_sq
    else:
        gamma_B_val = np.abs(h_B.conj() @ w)**2 / sigma_sq
        gamma_E_val = np.abs(h_E.conj() @ w)**2 / sigma_sq

    # 保密率R_s = [log2(1 + gamma_B) - log2(1 + gamma_E)]^+
    # 我们正在最大化log2((1+gamma_B)/(1+gamma_E))
    # 如果(1+gamma_E)为0(或非常小)且(1+gamma_B)不为0，速率可能会很大。
    # 分数F(w)的分子和分母是(sigma_sq + |h_x^H w|^2)/P_max
    # 所以比率是(sigma_sq + |h_B^H w|^2) / (sigma_sq + |h_E^H w|^2)
    
    numerator = sigma_sq + np.abs(h_B.conj() @ w)**2
    denominator = sigma_sq + np.abs(h_E.conj() @ w)**2

    if denominator < 1e-12: # 如果Eve没有接收到信号且噪声为零，避免除以零
        if numerator < 1e-12: # Bob也没有接收到信号
            return 0.0 # log2(1) = 0
        else: # Bob接收到信号，Eve + 噪声为零 -> 实际上是无限速率
            # 如果sigma_sq很小但为正，这种情况应该由大gamma_B处理
            # 对于目标值，这将非常大。保密率log2(large_val)。
            # 如果发生这种情况，让我们限制它或返回一个非常大的数字。
            # 当sigma_sq > 0时，分母总是> 0。
            # 这只在sigma_sq=0且h_E^H w = 0时发生。
            return np.log2(1 + np.abs(h_B.conj() @ w)**2 / 1e-12) # 一个非常大的速率

    ratio = numerator / denominator
    
    if ratio <= 0: # 如果sigma_sq > 0，不应该发生
        return -np.inf # 或一个非常小的数字，非正数的对数未定义
        
    rate = np.log2(ratio)
    # 问题P1有[C_B - C_E]^+。如果我们直接使用rate = C_B - C_E，它可能为负。
    # 论文中的优化在最大化分数时隐式地丢弃了[ ]^+，
    # 假设最优解将产生C_B > C_E。
    # 如果我们想匹配问题P1的目标，我们应该使用max(0, rate)。
    # 然而，为了比较*最大化能力*，原始的`rate`更好。
    return rate


def compare_with_random_simulation(h_B, h_E, P_max, sigma_sq, num_simulations=100000000):
    """
    将分析解与随机波束成形器的结果进行比较。

    参数:
        h_B, h_E, P_max, sigma_sq: 与optimize_beamformer_fixed_psi相同。
        num_simulations (int): 要测试的随机波束成形器数量。

    返回:
        tuple: (rate_analytical, w_analytical, max_rate_sim, w_best_sim)
    """
    N = h_B.shape[0]
    rng = np.random.default_rng()

    # 1. 分析解
    w_analytical = optimize_beamformer_fixed_psi(h_B, h_E, P_max, sigma_sq)
    rate_analytical = calculate_secrecy_rate(w_analytical, h_B, h_E, sigma_sq)

    # 2. 随机波束成形器模拟
    max_rate_sim = -np.inf
    w_best_sim = np.zeros(N, dtype=complex)

    for i in range(num_simulations):
        # 生成随机复向量
        real_part = rng.standard_normal(N)
        imag_part = rng.standard_normal(N)
        w_rand_unnormalized = real_part + 1j * imag_part
        
        # 归一化为单位范数
        norm_w_rand = np.linalg.norm(w_rand_unnormalized)
        if norm_w_rand < 1e-9: # 对于N > 0，极不可能
            w_rand_normalized = np.ones(N, dtype=complex) / np.sqrt(N)
        else:
            w_rand_normalized = w_rand_unnormalized / norm_w_rand
            
        # 缩放以满足功率约束
        w_rand = np.sqrt(P_max) * w_rand_normalized
        
        current_rate_sim = calculate_secrecy_rate(w_rand, h_B, h_E, sigma_sq)
        
        if current_rate_sim > max_rate_sim:
            max_rate_sim = current_rate_sim
            w_best_sim = w_rand
        
        if (i + 1) % (num_simulations // 10) == 0:
            print(f"模拟进度：{((i+1)/num_simulations*100):.0f}% 已完成...")

    return rate_analytical, w_analytical, max_rate_sim, w_best_sim


# --- 示例用法 ---
if __name__ == "__main__":
    N_antennas = 4
    P_max = 1.0
    sigma_sq = 0.01 # 确保sigma_sq > 0，使c > 0且B良好条件

    rng = np.random.default_rng(42)
    h_B_example = rng.normal(size=N_antennas) + 1j * rng.normal(size=N_antennas)
    # 使Eve的信道不同且可能更弱，以获得更清晰的最优解
    h_E_example = (rng.normal(size=N_antennas) + 1j * rng.normal(size=N_antennas)) * 0.5 
    # h_E_example = np.copy(h_B_example) * 0.1 # 测试用例：Eve非常弱且对齐
    # h_E_example = np.zeros_like(h_B_example) # 测试用例：Eve信道为空(B_matrix简化)

    print(f"N_antennas = {N_antennas}")
    print(f"P_max = {P_max}")
    print(f"sigma^2 = {sigma_sq}\n")
    print(f"h_B (示例) = {h_B_example}")
    print(f"h_E (示例) = {h_E_example}\n")

    # --- 直接调用优化器 ---
    w_optimal_direct = optimize_beamformer_fixed_psi(h_B_example, h_E_example, P_max, sigma_sq)
    rate_optimal_direct = calculate_secrecy_rate(w_optimal_direct, h_B_example, h_E_example, sigma_sq)
    print(f"--- 分析解 ---")
    print(f"最优波束成形器w* (直接) = {w_optimal_direct}")
    print(f"w*的功率 (直接): {np.linalg.norm(w_optimal_direct)**2:.6f}")
    print(f"实现的保密率 (直接): {rate_optimal_direct:.6f} bits/s/Hz\n")

    # --- 与模拟比较 ---
    num_sims = 10000 # 减少以加快测试，增加以获得更好的比较
    print(f"--- 运行模拟比较 (num_simulations = {num_sims}) ---")
    
    rate_analytical_comp, w_analytical_comp, max_rate_sim_comp, w_best_sim_comp = \
        compare_with_random_simulation(h_B_example, h_E_example, P_max, sigma_sq, num_simulations=num_sims)

    print(f"\n--- 比较结果 ---")
    print(f"分析解速率: {rate_analytical_comp:.6f} bits/s/Hz")
    # print(f"分析w*: {w_analytical_comp}")
    print(f"最大模拟速率 ({num_sims} 次尝试): {max_rate_sim_comp:.6f} bits/s/Hz")
    # print(f"最佳模拟w: {w_best_sim_comp}")

    if rate_analytical_comp >= max_rate_sim_comp - 1e-9: # 允许微小的数值差异
        print("\n成功：分析解优于或等于模拟中找到的最佳解。")
    else:
        print("\n警告：模拟找到了更好的解决方案。这表明分析求解器或理论应用可能存在问题。")
        print(f"差异 (分析 - 模拟): {rate_analytical_comp - max_rate_sim_comp}")

    # 进一步检查：w_analytical_comp的功率
    print(f"比较函数中分析w*的功率: {np.linalg.norm(w_analytical_comp)**2:.6f}")
    # 进一步检查：w_best_sim_comp的功率
    print(f"比较函数中最佳模拟w的功率: {np.linalg.norm(w_best_sim_comp)**2:.6f}")

    # 示例：如果sigma_sq非常小或为零会怎样？
    # print("\n--- 测试sigma_sq = 0 ---")
    # P_max_nz = 1.0
    # sigma_sq_zero = 0.0 # 这将触发警告/特殊处理
    # h_B_nz = np.array([1+1j, 0.5-0.5j])
    # h_E_nz = np.array([0.1+0.1j, 0.05-0.05j])
    # w_nz_sigma_zero = optimize_beamformer_fixed_psi(h_B_nz, h_E_nz, P_max_nz, sigma_sq_zero)
    # rate_nz_sigma_zero = calculate_secrecy_rate(w_nz_sigma_zero, h_B_nz, h_E_nz, sigma_sq_zero) # 将使用有效的小sigma
    # print(f"sigma_sq=0的w*: {w_nz_sigma_zero}")
    # print(f"sigma_sq=0的速率: {rate_nz_sigma_zero}")
    # print(f"注意：sigma_sq=0的速率在calculate_secrecy_rate中使用有效的小sigma_sq以避免log(inf)。")