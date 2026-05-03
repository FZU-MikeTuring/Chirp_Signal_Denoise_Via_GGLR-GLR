import matplotlib.pyplot as plt
import numpy as np
import torch

from transformer_model import Transformer


NORMALIZATION_SCALE = 50.0


def calculate_snr(clean_signal, noisy_signal, ignore_ratio=0.1):
    n = len(clean_signal)
    ignore = int(n * ignore_ratio)
    clean_mid = clean_signal[ignore:-ignore]
    noisy_mid = noisy_signal[ignore:-ignore]
    noise = noisy_mid - clean_mid
    signal_power = np.mean(clean_mid**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)


def generate_chirp_signal(fs=1000, T=1, f0=50, f1=200, a0=50, a1=1, sigma=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, T, int(fs * T), endpoint=False)
    k = (f1 - f0) / T
    ka = (a1 - a0) / T
    envelope = a0 + ka * t
    chirp_clean = envelope * np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2) + np.pi)

    noise = sigma * np.random.randn(len(t))
    chirp_noisy = chirp_clean + noise
    return t, chirp_clean, chirp_noisy


def denoise(model, noisy_signal, device="cuda"):
    model.eval()
    with torch.no_grad():
        noisy_tensor = (
            torch.FloatTensor(noisy_signal / NORMALIZATION_SCALE).unsqueeze(0).unsqueeze(-1).to(device)
        )
        denoised_tensor = model(src=noisy_tensor)
        denoised = denoised_tensor.squeeze().cpu().numpy() * NORMALIZATION_SCALE
    return denoised


if __name__ == "__main__":
    fs = 1000
    T = 1
    f0 = 50
    f1 = 200
    a0 = 50
    a1 = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 1
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_encoder_blocks = 6
    dropout = 0.1

    model = Transformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=num_encoder_blocks,
        dropout=dropout,
    )
    model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")
    print(f"Amplitude decay setting: a0={a0}, a1={a1}")

    sigma_list = [20, 10, 5, 2, 1, 0.5]
    random_seed = 42

    print("\n" + "=" * 60)
    print("Transformer denoising performance test")
    print("=" * 60)

    results = []

    for sigma_noise in sigma_list:
        print(f"\nCurrent noise std: {sigma_noise}")
        print("-" * 40)

        t, chirp_clean, chirp_noisy = generate_chirp_signal(
            fs, T, f0, f1, a0=a0, a1=a1, sigma=sigma_noise, seed=random_seed
        )

        chirp_denoised = denoise(model, chirp_noisy, device)

        snr_noisy = calculate_snr(chirp_clean, chirp_noisy)
        snr_denoised = calculate_snr(chirp_clean, chirp_denoised)

        print(f"Input SNR: {snr_noisy:.2f} dB")
        print(f"Output SNR: {snr_denoised:.2f} dB")
        print(f"SNR gain: {snr_denoised - snr_noisy:.2f} dB")

        results.append(
            {
                "sigma": sigma_noise,
                "snr_input": snr_noisy,
                "snr_output": snr_denoised,
                "snr_gain": snr_denoised - snr_noisy,
            }
        )

        if sigma_noise == 5:
            n_plot = min(len(t), len(chirp_clean), len(chirp_noisy), len(chirp_denoised))
            t_plot = t[:n_plot]
            chirp_clean_plot = chirp_clean[:n_plot]
            chirp_noisy_plot = chirp_noisy[:n_plot]
            chirp_denoised_plot = chirp_denoised[:n_plot]

            plt.figure(figsize=(14, 7))

            plt.subplot(2, 1, 1)
            plt.plot(t_plot, chirp_clean_plot, "g-", linewidth=1.2, label="Clean", alpha=0.8)
            plt.plot(
                t_plot,
                chirp_noisy_plot,
                "r-",
                linewidth=0.6,
                label=f"Noisy ($\\sigma$={sigma_noise})",
                alpha=0.6,
            )
            plt.title(f"(a) Clean vs Noisy Signal (Input SNR: {snr_noisy:.1f} dB)")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            error_noisy = chirp_noisy_plot - chirp_clean_plot
            error_denoised = chirp_denoised_plot - chirp_clean_plot
            plt.plot(
                t_plot,
                error_noisy,
                "r-",
                linewidth=0.5,
                alpha=0.5,
                label=f"Noisy Residual (Var: {np.var(error_noisy):.4f})",
            )
            plt.plot(
                t_plot,
                error_denoised,
                "b-",
                linewidth=0.8,
                label=f"Transformer Residual (Var: {np.var(error_denoised):.4f})",
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Residual Amplitude")
            plt.title(
                f"(b) Residual Comparison (Output SNR: {snr_denoised:.1f} dB, "
                f"Gain: {snr_denoised - snr_noisy:.1f} dB)"
            )
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("transformer_time_domain_result.png", dpi=300, bbox_inches="tight")
            plt.show()
            print("Saved figure: transformer_time_domain_result.png")

    print("\n" + "=" * 60)
    print("Transformer denoising summary")
    print("=" * 60)
    print(f"{'sigma':<12} {'input SNR(dB)':<15} {'output SNR(dB)':<15} {'gain(dB)':<12}")
    print("-" * 60)
    for r in results:
        print(f"{r['sigma']:<12} {r['snr_input']:<15.2f} {r['snr_output']:<15.2f} {r['snr_gain']:<12.2f}")
