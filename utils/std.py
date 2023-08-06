import numpy as np

def std_q(mean_g:float,std_g:float,mean_p:float,std_p:float,num_samples:int=10000):
    g_samples = np.random.normal(loc=mean_g, scale=std_g, size=num_samples)
    p_samples = np.random.normal(loc=mean_p, scale=std_p, size=num_samples)

    # 计算q的样本
    q_samples = g_samples / np.sqrt(1 - p_samples)
    q_samples = q_samples[~np.isnan(q_samples)]

    # 计算q的标准差
    std_q = np.std(q_samples)
    return std_q
