import torch
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from dcnn_model import DCNN

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

def generate_chirp_signal(fs=1000, T=1, f0=50, f1=200, sigma=None, seed=None):
    """生成Chirp信号"""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    k = (f1 - f0) / T
    chirp_clean = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2) + np.pi)
    
    noise = sigma * np.random.randn(len(t))
    chirp_noisy = chirp_clean + noise
    
    return t, chirp_clean, chirp_noisy

def denoise(model, noisy_signal, device='cuda'):
    """使用DCNN模型去噪"""
    model.eval()
    with torch.no_grad():
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(0).to(device)
        denoised_tensor = model(noisy_tensor)
        denoised = denoised_tensor.squeeze().cpu().numpy()
    return denoised

if __name__ == "__main__":
    # 参数设置
    fs = 1000
    T = 1
    f0 = 50
    f1 = 200
    N = int(fs * T)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    model = DCNN(input_channels=1, N=N)
    model.load_state_dict(torch.load('dcnn_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型加载成功，使用设备: {device}")
    
    # 测试的噪声标准差
    sigma_list = [2, 0.4, 0.2, 0.1, 0.05, 0.025]
    random_seed = 42
    
    print("\n" + "="*60)
    print("DCNN方法去噪性能测试")
    print("="*60)
    
    results = []
    
    for sigma_noise in sigma_list:
        print(f"\n当前噪声标准差: {sigma_noise}")
        print("-"*40)
        
        # 生成信号
        t, chirp_clean, chirp_noisy = generate_chirp_signal(
            fs, T, f0, f1, sigma=sigma_noise, seed=random_seed
        )
        
        # DCNN去噪
        chirp_denoised = denoise(model, chirp_noisy, device)
        
        # 计算SNR
        snr_noisy = calculate_snr(chirp_clean, chirp_noisy)
        snr_denoised = calculate_snr(chirp_clean, chirp_denoised)
        
        print(f"去噪前SNR: {snr_noisy:.2f} dB")
        print(f"去噪后SNR: {snr_denoised:.2f} dB")
        print(f"SNR提升: {snr_denoised - snr_noisy:.2f} dB")
        
        results.append({
            'sigma': sigma_noise,
            'snr_input': snr_noisy,
            'snr_output': snr_denoised,
            'snr_gain': snr_denoised - snr_noisy
        })
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("DCNN方法去噪性能汇总")
    print("="*60)
    print(f"{'标准差':<12} {'输入SNR(dB)':<15} {'输出SNR(dB)':<15} {'提升(dB)':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['sigma']:<12} {r['snr_input']:<15.2f} {r['snr_output']:<15.2f} {r['snr_gain']:<12.2f}")