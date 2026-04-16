import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_model import Transformer

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
    """使用Transformer编码器去噪"""
    model.eval()
    with torch.no_grad():
        # 输入形状: (1, N, 1)
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(-1).to(device)
        denoised_tensor = model(src=noisy_tensor)
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
    
    # Transformer参数（必须与训练时一致）
    input_dim = 1
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_encoder_blocks = 6
    dropout = 0.1
    
    # 加载模型
    model = Transformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=0,
        dropout=dropout,
        encoder_only=True
    )
    model.load_state_dict(torch.load('transformer_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型加载成功，使用设备: {device}")
    
    # 测试的噪声标准差
    sigma_list = [2, 0.4, 0.2, 0.1, 0.05, 0.025]
    random_seed = 42
    
    print("\n" + "="*60)
    print("Transformer编码器去噪性能测试")
    print("="*60)
    
    results = []
    
    for sigma_noise in sigma_list:
        print(f"\n当前噪声标准差: {sigma_noise}")
        print("-"*40)
        
        # 生成信号
        t, chirp_clean, chirp_noisy = generate_chirp_signal(
            fs, T, f0, f1, sigma=sigma_noise, seed=random_seed
        )
        
        # Transformer去噪
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
        
        # ========== 只在 sigma=0.2 时绘图 ==========
        if sigma_noise == 0.2:
            print(f"\n正在绘制 sigma={sigma_noise} 的时域对比图...")
            
            # 绘图数据准备
            n_plot = min(len(t), len(chirp_clean), len(chirp_noisy), len(chirp_denoised))
            t_plot = t[:n_plot]
            chirp_clean_plot = chirp_clean[:n_plot]
            chirp_noisy_plot = chirp_noisy[:n_plot]
            chirp_denoised_plot = chirp_denoised[:n_plot]
            
            # 创建图形
            plt.figure(figsize=(14, 10))
            
            # 子图1：干净信号 vs 含噪信号
            plt.subplot(3, 1, 1)
            plt.plot(t_plot, chirp_clean_plot, 'g-', linewidth=1.2, label='Clean', alpha=0.8)
            plt.plot(t_plot, chirp_noisy_plot, 'r-', linewidth=0.6, label=f'Noisy ($\\sigma$={sigma_noise})', alpha=0.6)
            plt.title(f'(a) Original vs Noisy Signal (Input SNR: {snr_noisy:.1f} dB)')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # 子图2：干净信号 vs Transformer去噪信号
            plt.subplot(3, 1, 2)
            plt.plot(t_plot, chirp_clean_plot, 'g-', linewidth=1.2, label='Clean', alpha=0.8)
            plt.plot(t_plot, chirp_denoised_plot, 'b-', linewidth=1, label='Denoised (Transformer)')
            plt.title(f'(b) Original vs Denoised Signal (Output SNR: {snr_denoised:.1f} dB, Gain: {snr_denoised-snr_noisy:.1f} dB)')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            # 子图3：去噪前后误差对比
            plt.subplot(3, 1, 3)
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
            plt.savefig('transformer_time_domain_result.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("图像已保存为: transformer_time_domain_result.png")
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("Transformer编码器去噪性能汇总")
    print("="*60)
    print(f"{'标准差':<12} {'输入SNR(dB)':<15} {'输出SNR(dB)':<15} {'提升(dB)':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['sigma']:<12} {r['snr_input']:<15.2f} {r['snr_output']:<15.2f} {r['snr_gain']:<12.2f}")