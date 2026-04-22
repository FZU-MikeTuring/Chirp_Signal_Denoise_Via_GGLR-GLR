from pathlib import Path
import csv
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.signal import hilbert
from scipy.sparse import diags, lil_matrix


ROOT_DIR = Path(__file__).resolve().parent
GGLR_DIR = ROOT_DIR / "GGLR-GLR"
DCNN_DIR = ROOT_DIR / "DCNN"
TRANSFORMER_DIR = ROOT_DIR / "Transformer"
OUTPUT_DIR = ROOT_DIR / "generated_figures"


def load_symbol_from_file(module_name, file_path, symbol_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module '{module_name}' from '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(f"Module '{module_name}' does not define '{symbol_name}'.") from exc


DCNN = load_symbol_from_file("dcnn_model", DCNN_DIR / "dcnn_model.py", "DCNN")
Transformer = load_symbol_from_file(
    "transformer_model",
    TRANSFORMER_DIR / "transformer_model.py",
    "Transformer",
)


def generate_chirp_signal(fs=1000, T=1.0, f0=50.0, f1=200.0, sigma=0.2, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    k = (f1 - f0) / T
    chirp_clean = np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2) + np.pi)
    chirp_noisy = chirp_clean + sigma * rng.standard_normal(len(t))
    return t, chirp_clean, chirp_noisy


def calculate_snr(clean_signal, test_signal, ignore_ratio=0.1):
    n = len(clean_signal)
    ignore = int(n * ignore_ratio)
    clean_mid = clean_signal[ignore:-ignore]
    test_mid = test_signal[ignore:-ignore]
    noise = test_mid - clean_mid
    signal_power = np.mean(clean_mid**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)


def glr(amplitude, sigma=0.1, lambda_reg=600):
    n = len(amplitude)
    neighbors_count = n // 2
    weights = lil_matrix((n, n))

    for i in range(n):
        distances = np.abs(amplitude - amplitude[i]).flatten()
        neighbors = np.argsort(distances)[1:neighbors_count + 1]
        for j in neighbors:
            weight = np.exp(-(distances[j] ** 2) / (2 * sigma**2))
            weights[i, j] = weight
            weights[j, i] = weight

    degree = diags(np.array(weights.sum(axis=1)).flatten(), 0)
    laplacian = degree - weights.tocsr()
    system = np.eye(n) + lambda_reg * laplacian
    return np.linalg.solve(system, amplitude)


def gglr(frequency, sigma=600, lambda_reg=0.1):
    n = len(frequency)
    grad_n = n - 1
    neighbors_count = grad_n // 2

    grad_matrix = lil_matrix((grad_n, n))
    for i in range(grad_n):
        grad_matrix[i, i] = -1
        grad_matrix[i, i + 1] = 1

    grad_frequency = frequency[1:] - frequency[:-1]
    weights = lil_matrix((grad_n, grad_n))

    for i in range(grad_n):
        distances = np.abs(grad_frequency - grad_frequency[i]).flatten()
        neighbors = np.argsort(distances)[1:neighbors_count + 1]
        for j in neighbors:
            weight = np.exp(-(distances[j] ** 2) / (2 * sigma**2))
            weights[i, j] = weight
            weights[j, i] = weight

    degree = diags(np.array(weights.sum(axis=1)).flatten(), 0)
    laplacian = degree - weights.tocsr()
    smoothness = grad_matrix.T @ laplacian @ grad_matrix
    system = np.eye(n) + lambda_reg * smoothness
    return np.linalg.solve(system, frequency)


def proposed_denoise(chirp_noisy, fs, sigmaa=0.1, lambdaa=600, sigmaf=600, lambdaf=0.1):
    analytic_noisy = hilbert(chirp_noisy)
    amplitude_noisy = np.abs(analytic_noisy)
    phase_noisy = np.unwrap(np.angle(analytic_noisy))
    init_phase = phase_noisy[0]
    frequency_noisy = np.diff(phase_noisy) / (2 * np.pi) * fs

    amplitude_restored = glr(amplitude_noisy, sigma=sigmaa, lambda_reg=lambdaa)
    frequency_restored = gglr(frequency_noisy, sigma=sigmaf, lambda_reg=lambdaf)

    phase_increment = np.cumsum(frequency_restored) * (2 * np.pi / fs)
    phase_reconstructed = np.full(len(phase_increment) + 1, init_phase)
    phase_reconstructed[1:] += phase_increment

    chirp_denoised = np.real(amplitude_restored * np.exp(1j * phase_reconstructed))

    return {
        "analytic_noisy": analytic_noisy,
        "amplitude_noisy": amplitude_noisy,
        "frequency_noisy": frequency_noisy,
        "amplitude_restored": amplitude_restored,
        "frequency_restored": frequency_restored,
        "chirp_denoised": chirp_denoised,
    }


def extract_clean_features(chirp_clean, fs):
    analytic_clean = hilbert(chirp_clean)
    amplitude_clean = np.abs(analytic_clean)
    phase_clean = np.unwrap(np.angle(analytic_clean))
    frequency_clean = np.diff(phase_clean) / (2 * np.pi) * fs
    return {
        "amplitude_clean": amplitude_clean,
        "frequency_clean": frequency_clean,
    }


def load_dcnn(device, signal_length):
    model = DCNN(input_channels=1, N=signal_length)
    model.load_state_dict(torch.load(DCNN_DIR / "dcnn_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def dcnn_denoise(model, noisy_signal, device):
    with torch.no_grad():
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(0).to(device)
        denoised_tensor = model(noisy_tensor)
    return denoised_tensor.squeeze().cpu().numpy()


def load_transformer(device):
    model = Transformer(
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_encoder_blocks=6,
        num_decoder_blocks=0,
        dropout=0.1,
        encoder_only=True,
    )
    model.load_state_dict(torch.load(TRANSFORMER_DIR / "transformer_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def transformer_denoise(model, noisy_signal, device):
    with torch.no_grad():
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(-1).to(device)
        denoised_tensor = model(src=noisy_tensor)
    return denoised_tensor.squeeze().cpu().numpy()


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup_style():
    plt.style.use("default")
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14


def plot_framework():
    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")

    boxes = [
        (0.03, 0.3, 0.15, 0.4, "Noisy chirp"),
        (0.22, 0.3, 0.16, 0.4, "Hilbert\ntransform"),
        (0.42, 0.55, 0.2, 0.22, "Instantaneous\namplitude"),
        (0.42, 0.23, 0.2, 0.22, "Instantaneous\nfrequency"),
        (0.68, 0.55, 0.12, 0.22, "GLR"),
        (0.68, 0.23, 0.12, 0.22, "GGLR"),
        (0.84, 0.3, 0.13, 0.4, "Signal\nreconstruction"),
    ]

    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.5,
            edgecolor="#274c77",
            facecolor="#d9eaf7",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    arrows = [
        ((0.18, 0.5), (0.22, 0.5)),
        ((0.38, 0.57), (0.42, 0.66)),
        ((0.38, 0.43), (0.42, 0.34)),
        ((0.62, 0.66), (0.68, 0.66)),
        ((0.62, 0.34), (0.68, 0.34)),
        ((0.80, 0.66), (0.84, 0.57)),
        ((0.80, 0.34), (0.84, 0.43)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=14, linewidth=1.6, color="#274c77"))

    ax.set_title("Overall framework of the proposed Chirp denoising method.")
    save_figure(fig, "fig1_framework.png")


def plot_observation_and_restoration_figures(t, clean_features, proposed_features):
    t_freq = t[: len(clean_features["frequency_clean"])]
    t_amp = t[: len(clean_features["amplitude_clean"])]

    frequency_clean = clean_features["frequency_clean"]
    amplitude_clean = clean_features["amplitude_clean"]
    frequency_noisy = proposed_features["frequency_noisy"][: len(frequency_clean)]
    amplitude_noisy = proposed_features["amplitude_noisy"][: len(amplitude_clean)]
    frequency_restored = proposed_features["frequency_restored"][: len(frequency_clean)]
    amplitude_restored = proposed_features["amplitude_restored"][: len(amplitude_clean)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_freq, frequency_clean, color="#2a9d8f", linewidth=1.6, label="Ground truth frequency")
    ax.plot(t_freq, frequency_noisy, color="#e76f51", linewidth=1.0, alpha=0.8, label="Noisy extracted frequency")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Instantaneous frequency extracted by the Hilbert transform from a noisy Chirp signal.")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig2_hilbert_frequency_observation.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_amp, amplitude_clean, color="#2a9d8f", linewidth=1.6, label="Ground truth amplitude")
    ax.plot(t_amp, amplitude_noisy, color="#e76f51", linewidth=1.0, alpha=0.8, label="Noisy extracted amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude envelope extracted by the Hilbert transform from a noisy Chirp signal.")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig3_hilbert_amplitude_observation.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_amp, amplitude_noisy, color="#e76f51", linewidth=0.9, alpha=0.7, label="Noisy amplitude")
    ax.plot(t_amp, amplitude_restored, color="#1d3557", linewidth=1.5, label="GLR-restored amplitude")
    ax.plot(t_amp, amplitude_clean, color="#2a9d8f", linewidth=1.4, linestyle="--", label="Ground truth amplitude")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude restoration using GLR.")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig4_glr_amplitude_restoration.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_freq, frequency_noisy, color="#e76f51", linewidth=0.9, alpha=0.7, label="Noisy frequency")
    ax.plot(t_freq, frequency_restored, color="#1d3557", linewidth=1.5, label="GGLR-restored frequency")
    ax.plot(t_freq, frequency_clean, color="#2a9d8f", linewidth=1.4, linestyle="--", label="Ground truth frequency")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Instantaneous frequency restoration using GGLR.")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig5_gglr_frequency_restoration.png")


def plot_time_domain_result(t, clean_signal, noisy_signal, denoised_signal, method_label, filename, snr_in, snr_out):
    residual_noisy = noisy_signal - clean_signal
    residual_denoised = denoised_signal - clean_signal

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, clean_signal, color="#2a9d8f", linewidth=1.3, label="Clean")
    axes[0].plot(t, noisy_signal, color="#e76f51", linewidth=0.8, alpha=0.7, label="Noisy")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"(a) Clean vs noisy signal (Input SNR: {snr_in:.2f} dB)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, residual_noisy, color="#e76f51", linewidth=0.8, alpha=0.6, label="Noisy residual")
    axes[1].plot(t, residual_denoised, color="#1d3557", linewidth=1.0, label=f"{method_label} residual")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"(b) Residual comparison (Output SNR: {snr_out:.2f} dB, Gain: {snr_out - snr_in:.2f} dB)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Time-domain denoising results of the {method_label} method.")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_all_methods_comparison(t, clean_signal, noisy_signal, denoised_signals):
    residual_noisy = noisy_signal - clean_signal

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t, clean_signal, color="#2a9d8f", linewidth=1.3, label="Original")
    axes[0].plot(t, noisy_signal, color="#e76f51", linewidth=0.8, alpha=0.7, label="Noisy")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("(a) Original and noisy signals")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    colors = {
        "Proposed": "#1d3557",
        "DCNN": "#6a994e",
        "Transformer": "#7b2cbf",
    }
    axes[1].plot(t, residual_noisy, color="#e76f51", linewidth=0.8, alpha=0.4, label="Noisy residual")
    for method_name, denoised_signal in denoised_signals.items():
        axes[1].plot(
            t,
            denoised_signal - clean_signal,
            color=colors[method_name],
            linewidth=1.0,
            label=f"{method_name} residual",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("(b) Residual comparison of different denoising methods")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Time-domain comparison of different denoising methods under the same noise condition.")
    fig.tight_layout()
    save_figure(fig, "fig8_all_methods_time_domain_comparison.png")


def plot_quantitative_figures(results):
    sigma_values = [row["sigma"] for row in results["Proposed"]]
    method_colors = {
        "Proposed": "#1d3557",
        "DCNN": "#6a994e",
        "Transformer": "#7b2cbf",
    }

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for method_name, rows in results.items():
        ax.plot(
            sigma_values,
            [row["snr_output"] for row in rows],
            marker="o",
            linewidth=1.8,
            color=method_colors[method_name],
            label=method_name,
        )
    ax.set_xlabel("Input noise standard deviation")
    ax.set_ylabel("Output SNR (dB)")
    ax.set_title("Output SNR versus input noise standard deviation for different methods.")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, "fig9_output_snr_vs_sigma.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for method_name, rows in results.items():
        ax.plot(
            sigma_values,
            [row["snr_gain"] for row in rows],
            marker="o",
            linewidth=1.8,
            color=method_colors[method_name],
            label=method_name,
        )
    ax.set_xlabel("Input noise standard deviation")
    ax.set_ylabel("SNR improvement (dB)")
    ax.set_title("SNR improvement versus input noise standard deviation for different methods.")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, "fig10_snr_improvement_vs_sigma.png")


def plot_lambda_sensitivity(fs, T, f0, f1, sigma_noise, seed, ignore_ratio):
    lambda_values = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    _, chirp_clean, chirp_noisy = generate_chirp_signal(fs, T, f0, f1, sigma=sigma_noise, seed=seed)

    snr_output = []
    snr_gain = []
    for lambda_reg in lambda_values:
        denoised = proposed_denoise(
            chirp_noisy,
            fs,
            sigmaa=0.1,
            lambdaa=600,
            sigmaf=600,
            lambdaf=lambda_reg,
        )["chirp_denoised"]
        snr_in = calculate_snr(chirp_clean, chirp_noisy, ignore_ratio=ignore_ratio)
        snr_out = calculate_snr(chirp_clean, denoised, ignore_ratio=ignore_ratio)
        snr_output.append(snr_out)
        snr_gain.append(snr_out - snr_in)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    ax1.semilogx(lambda_values, snr_output, marker="o", color="#1d3557", linewidth=1.8, label="Output SNR")
    ax1.set_xlabel("Regularization parameter $\\lambda$")
    ax1.set_ylabel("Output SNR (dB)", color="#1d3557")
    ax1.tick_params(axis="y", labelcolor="#1d3557")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.semilogx(lambda_values, snr_gain, marker="s", color="#e76f51", linewidth=1.5, label="SNR improvement")
    ax2.set_ylabel("SNR improvement (dB)", color="#e76f51")
    ax2.tick_params(axis="y", labelcolor="#e76f51")

    ax1.set_title("Effect of regularization parameter $\\lambda$ on denoising performance.")
    save_figure(fig, "fig11_lambda_sensitivity.png")


def export_metrics(results):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "metrics_summary.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "sigma", "snr_input", "snr_output", "snr_gain"])
        for method_name, rows in results.items():
            for row in rows:
                writer.writerow([method_name, row["sigma"], row["snr_input"], row["snr_output"], row["snr_gain"]])


def main():
    setup_style()

    fs = 1000
    T = 1.0
    f0 = 50.0
    f1 = 200.0
    sigma_focus = 0.2
    sigma_list = [2, 0.4, 0.2, 0.1, 0.05, 0.025]
    seed = 42
    ignore_ratio = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    plot_framework()

    t, chirp_clean, chirp_noisy = generate_chirp_signal(fs, T, f0, f1, sigma=sigma_focus, seed=seed)
    clean_features = extract_clean_features(chirp_clean, fs)
    proposed_features = proposed_denoise(chirp_noisy, fs)
    chirp_proposed = proposed_features["chirp_denoised"]

    dcnn_model = load_dcnn(device, len(chirp_clean))
    transformer_model = load_transformer(device)
    chirp_dcnn = dcnn_denoise(dcnn_model, chirp_noisy, device)
    chirp_transformer = transformer_denoise(transformer_model, chirp_noisy, device)

    plot_observation_and_restoration_figures(t, clean_features, proposed_features)

    snr_input_focus = calculate_snr(chirp_clean, chirp_noisy, ignore_ratio=ignore_ratio)
    plot_time_domain_result(
        t,
        chirp_clean,
        chirp_noisy,
        chirp_proposed,
        "proposed",
        "fig6_proposed_time_domain_result.png",
        snr_input_focus,
        calculate_snr(chirp_clean, chirp_proposed, ignore_ratio=ignore_ratio),
    )
    plot_time_domain_result(
        t,
        chirp_clean,
        chirp_noisy,
        chirp_dcnn,
        "DCNN",
        "fig7_dcnn_time_domain_result.png",
        snr_input_focus,
        calculate_snr(chirp_clean, chirp_dcnn, ignore_ratio=ignore_ratio),
    )
    plot_time_domain_result(
        t,
        chirp_clean,
        chirp_noisy,
        chirp_transformer,
        "Transformer",
        "fig8_transformer_time_domain_result.png",
        snr_input_focus,
        calculate_snr(chirp_clean, chirp_transformer, ignore_ratio=ignore_ratio),
    )
    plot_all_methods_comparison(
        t,
        chirp_clean,
        chirp_noisy,
        {
            "Proposed": chirp_proposed,
            "DCNN": chirp_dcnn,
            "Transformer": chirp_transformer,
        },
    )

    results = {"Proposed": [], "DCNN": [], "Transformer": []}
    for sigma_noise in sigma_list:
        _, chirp_clean_sigma, chirp_noisy_sigma = generate_chirp_signal(fs, T, f0, f1, sigma=sigma_noise, seed=seed)
        chirp_proposed_sigma = proposed_denoise(chirp_noisy_sigma, fs)["chirp_denoised"]
        chirp_dcnn_sigma = dcnn_denoise(dcnn_model, chirp_noisy_sigma, device)
        chirp_transformer_sigma = transformer_denoise(transformer_model, chirp_noisy_sigma, device)

        snr_input = calculate_snr(chirp_clean_sigma, chirp_noisy_sigma, ignore_ratio=ignore_ratio)
        method_outputs = {
            "Proposed": chirp_proposed_sigma,
            "DCNN": chirp_dcnn_sigma,
            "Transformer": chirp_transformer_sigma,
        }
        for method_name, denoised_signal in method_outputs.items():
            snr_output = calculate_snr(chirp_clean_sigma, denoised_signal, ignore_ratio=ignore_ratio)
            results[method_name].append(
                {
                    "sigma": sigma_noise,
                    "snr_input": snr_input,
                    "snr_output": snr_output,
                    "snr_gain": snr_output - snr_input,
                }
            )

    plot_quantitative_figures(results)
    plot_lambda_sensitivity(fs, T, f0, f1, sigma_focus, seed, ignore_ratio)
    export_metrics(results)

    print(f"All figures have been saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
