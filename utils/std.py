import numpy as np

def std_q(mean_g:float,std_g:float,mean_p:float,std_p:float,num_samples:int=10000):
    g_samples = np.random.normal(loc=mean_g, scale=std_g, size=num_samples)
    p_samples = np.random.normal(loc=mean_p, scale=std_p, size=num_samples)

    g_samples = g_samples[(g_samples > 0) & (g_samples < 1)]
    p_samples = p_samples[(p_samples > 0) & (p_samples < 1)]

    # 找出两者中最小的长度，并截取两者使其长度相等
    min_len = min(len(g_samples), len(p_samples))
    g_samples = g_samples[:min_len]
    p_samples = p_samples[:min_len]

    # 计算q的样本
    q_samples = g_samples / np.sqrt(1 - p_samples)
    q_samples = q_samples[~np.isnan(q_samples)]

    # 计算q的标准差
    std_q = np.std(q_samples)
    if np.isnan(std_q) or np.isinf(std_q):
        return 0
    else:
        return std_q
