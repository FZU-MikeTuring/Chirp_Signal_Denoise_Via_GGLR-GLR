import numpy as np
from scipy.signal import hilbert
from scipy.sparse import diags, lil_matrix
import matplotlib.pylab as plt

# 生成Chirp信号
def generate_chirp_signal(fs=1000, T=1, f0=50, f1=200, sigma=None, seed=None):
    """
    生成Chirp信号
    sigma: 高斯白噪声的标准差
    seed: 随机种子，用于可复现的噪声生成
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    k = (f1 - f0) / T
    chirp_clean = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2) + np.pi)
    
    # 生成高斯白噪声（直接使用给定的标准差）
    noise = sigma * np.random.randn(len(t))
    print(f"加入噪声的标准差为{sigma:.4f}\n")
    chirp_noisy = chirp_clean + noise
    
    return t, chirp_clean, chirp_noisy

def calculate_snr(clean_signal, noisy_signal, ignore_ratio=0.1):
    """计算信噪比，忽略边缘效应"""
    n = len(clean_signal)
    ignore = int(n * ignore_ratio)
    clean_mid = clean_signal[ignore:-ignore]
    noisy_mid = noisy_signal[ignore:-ignore]
    noise = noisy_mid - clean_mid
    signal_power = np.mean(clean_mid**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

def GLR(am, sigma=50, lambda_reg=1):
    """对幅度进行图平滑处理"""
    M = len(am)
    K = M//2
    W = lil_matrix((M, M))
    
    for i in range(M):
        distances = np.abs(am - am[i]).flatten()
        neighbors = np.argsort(distances)[1:K+1]
        for j in neighbors:
            weight = np.exp(-distances[j]**2 / (2 * sigma**2))
            W[i, j] = weight
            W[j, i] = weight
    
    D = diags(np.array(W.sum(axis=1)).flatten(), 0)
    L = D - W.tocsr()
    I = np.eye(M)
    A = I + lambda_reg * L
    return np.linalg.solve(A, am)

def GGLR(freq, sigma=50, lambda_reg=0.1):
    """GGLR恢复频率"""
    M = len(freq)
    grad_M = M - 1
    K = grad_M // 2
    F_h = lil_matrix((grad_M, M))
    for i in range(grad_M):
        F_h[i, i] = -1
        F_h[i, i+1] = 1
    
    grad_f = freq[1:] - freq[:-1]
    W = lil_matrix((grad_M, grad_M))
    
    for i in range(grad_M):
        distances = np.abs(grad_f - grad_f[i]).flatten()
        neighbors = np.argsort(distances)[1:K+1]
        for j in neighbors:
            weight = np.exp(-distances[j]**2 / (2 * sigma**2))
            W[i, j] = weight
            W[j, i] = weight
    
    D = diags(np.array(W.sum(axis=1)).flatten(), 0)
    L = D - W.tocsr()
    LL = F_h.T @ L @ F_h
    I = np.eye(M)
    A = I + lambda_reg * LL
    return np.linalg.solve(A, freq)

# ============ 参数设置 ============
fs = 1000
T = 1
f0 = 50
f1 = 200
ignore_ratio = 0.1
random_seed = 42

# 噪声标准差列表
sigma_list = [2, 0.4, 0.2, 0.1, 0.05, 0.025]

# GGLR/GLR参数
sigmaf = 600
lambdaf = 0.1
sigmaa = 0.1
lambdaa = 600
# ================================

np.random.seed(random_seed)

for sigma_noise in sigma_list:
    print(f"\n{'='*50}")
    print(f"当前噪声标准差: {sigma_noise}")
    print(f"{'='*50}")
    
    # 生成信号
    t, chirp_clean, chirp_noisy = generate_chirp_signal(fs, T, f0, f1, sigma=sigma_noise, seed=random_seed)
    
    # 提取特征
    z_t = hilbert(chirp_noisy)
    a = np.abs(z_t)
    phase = np.unwrap(np.angle(z_t))
    init_phase = phase[0]
    freq = np.diff(phase) / (2 * np.pi) * fs
    
    # 干净信号特征
    z_clean = hilbert(chirp_clean)
    a_clean = np.abs(z_clean)
    freq_clean = np.diff(np.unwrap(np.angle(z_clean))) / (2 * np.pi) * fs
    
    # 先对幅度进行GLR，再对频率进行GGLR
    ad = GLR(a, sigmaa, lambdaa)
    fd = GGLR(freq, sigmaf, lambdaf)
    
    # 恢复时域信号
    phase_increments = np.cumsum(fd) * (2 * np.pi / fs)
    phase_reconstructed = np.full(len(phase_increments) + 1, init_phase)
    phase_reconstructed[1:] += phase_increments
    analytic = ad * np.exp(1j * phase_reconstructed)
    chirp_denoised = np.real(analytic)
    
    # 计算SNR
    snr_noisy = calculate_snr(chirp_clean, chirp_noisy, ignore_ratio)
    snr_denoised = calculate_snr(chirp_clean, chirp_denoised, ignore_ratio)
    
    print(f"去噪前SNR: {snr_noisy:.2f} dB")
    print(f"去噪后SNR: {snr_denoised:.2f} dB")
    print(f"SNR提升: {snr_denoised - snr_noisy:.2f} dB")
    
    # 只在 sigma=0.2 时绘图
    if sigma_noise == 0.2:
        # 绘图数据准备
        n_plot = min(len(t), len(chirp_clean), len(chirp_noisy), len(chirp_denoised))
        t_plot = t[:n_plot]
        chirp_clean_plot = chirp_clean[:n_plot]
        chirp_noisy_plot = chirp_noisy[:n_plot]
        chirp_denoised_plot = chirp_denoised[:n_plot]
        
        n_freq = min(len(freq_clean), len(freq), len(fd))
        t_freq = t[:n_freq]
        freq_clean_plot = freq_clean[:n_freq]
        freq_plot = freq[:n_freq]
        fd_plot = fd[:n_freq]
        
        n_amp = min(len(a_clean), len(a), len(ad))
        t_amp = t[:n_amp]
        a_clean_plot = a_clean[:n_amp]
        a_plot = a[:n_amp]
        ad_plot = ad[:n_amp]
        
        freq_error_noisy = freq_plot - freq_clean_plot
        freq_error_denoised = fd_plot - freq_clean_plot
        
        # ========== Figure 1: 时域去噪效果对比 ==========
        plt.figure(figsize=(14, 10))
        
        plt.subplot(3,1,1)
        plt.plot(t_plot, chirp_clean_plot, 'g-', linewidth=1.2, label='Clean', alpha=0.8)
        plt.plot(t_plot, chirp_noisy_plot, 'r-', linewidth=0.6, label=f'Noisy ($\\sigma$={sigma_noise})', alpha=0.6)
        plt.title(f'(a) Original vs Noisy Signal (Input SNR: {snr_noisy:.1f} dB)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3,1,2)
        plt.plot(t_plot, chirp_clean_plot, 'g-', linewidth=1.2, label='Clean', alpha=0.8)
        plt.plot(t_plot, chirp_denoised_plot, 'b-', linewidth=1, label='Denoised (Ours)')
        plt.title(f'(b) Original vs Denoised Signal (Output SNR: {snr_denoised:.1f} dB, Gain: {snr_denoised-snr_noisy:.1f} dB)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3,1,3)
        error_noisy = chirp_noisy_plot - chirp_clean_plot
        error_denoised = chirp_denoised_plot - chirp_clean_plot
        plt.plot(t_plot, error_noisy, 'r-', linewidth=0.5, alpha=0.5, label=f'Noisy Error (Var: {np.var(error_noisy):.4f})')
        plt.plot(t_plot, error_denoised, 'b-', linewidth=0.8, label=f'Denoised Error (Var: {np.var(error_denoised):.4f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Error Amplitude')
        plt.title('(c) Reconstruction Error Comparison')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time_domain_result.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
        # ========== Figure 2: 时频域恢复效果 ==========
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2,2,1)
        plt.plot(t_freq, freq_clean_plot, 'g-', linewidth=1.5, label='Ground Truth')
        plt.plot(t_freq, fd_plot, 'b-', linewidth=1.2, label='GGLR Recovered')
        plt.title(f'(a) Instantaneous Frequency Recovery (Error STD: {np.std(freq_error_denoised):.2f} Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2,2,2)
        plt.hist(freq_error_noisy, bins=40, alpha=0.5, color='red', label=f'Noisy (STD={np.std(freq_error_noisy):.2f})')
        plt.hist(freq_error_denoised, bins=40, alpha=0.5, color='blue', label=f'Recovered (STD={np.std(freq_error_denoised):.2f})')
        plt.xlabel('Frequency Error (Hz)')
        plt.ylabel('Count')
        plt.title('(b) Frequency Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2,2,3)
        plt.plot(t_amp, a_clean_plot, 'g-', linewidth=1.5, label='Ground Truth')
        plt.plot(t_amp, ad_plot, 'b-', linewidth=1.2, label='GLR Smoothed')
        plt.title(f'(c) Amplitude Envelope Recovery (MSE: {np.mean((ad_plot - a_clean_plot)**2):.4f})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2,2,4)
        plt.plot(t_amp, a_plot - a_clean_plot, 'r-', linewidth=0.6, alpha=0.5, label='Noisy Error')
        plt.plot(t_amp, ad_plot - a_clean_plot, 'b-', linewidth=0.8, label='Smoothed Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude Error')
        plt.title('(d) Amplitude Error Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('frequency_amplitude_result.pdf', dpi=300, bbox_inches='tight')
        plt.show()