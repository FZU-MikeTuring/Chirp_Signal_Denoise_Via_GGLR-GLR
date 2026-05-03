from pathlib import Path
import csv
import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT_DIR = Path(__file__).resolve().parent
GGLR_DIR = ROOT_DIR / "GGLR"
DCNN_DIR = ROOT_DIR / "DCNN"
TRANSFORMER_DIR = ROOT_DIR / "Transformer"
OUTPUT_DIR = ROOT_DIR / "generated_figures"


def hilbert_transform(signal):
    signal = np.asarray(signal)
    n = signal.shape[0]
    spectrum = np.fft.fft(signal)
    multiplier = np.zeros(n)
    if n % 2 == 0:
        multiplier[0] = 1.0
        multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier)


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
generate_chirp_signal_core = load_symbol_from_file("gglr_module", GGLR_DIR / "GGLR.py", "generate_chirp_signal")
calculate_snr = load_symbol_from_file("gglr_module", GGLR_DIR / "GGLR.py", "calculate_snr")
denoise_core = load_symbol_from_file("gglr_module", GGLR_DIR / "GGLR.py", "denoise")


def generate_chirp_signal(fs=1000, T=1.0, f0=50.0, f1=200.0, a0=50.0, a1=1.0, sigma=0.2, seed=42):
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    chirp_clean, chirp_noisy = generate_chirp_signal_core(
        fs=fs,
        T=T,
        f0=f0,
        f1=f1,
        a0=a0,
        a1=a1,
        sigma=sigma,
        seed=seed,
    )
    return t, chirp_clean, chirp_noisy


def proposed_denoise(chirp_noisy, fs, sigmaa=600, lambdaa=100, sigmaf=600, lambdaf=0.1, epochs=3):
    analytic_observed = hilbert_transform(chirp_noisy)
    amplitude_noisy = np.abs(analytic_observed)
    phase_noisy = np.unwrap(np.angle(analytic_observed))
    frequency_noisy = np.diff(phase_noisy) / (2 * np.pi) * fs

    current_signal = denoise_core(
        chirp_noisy,
        fs,
        sigmaa=sigmaa,
        lambdaa=lambdaa,
        sigmaf=sigmaf,
        lambdaf=lambdaf,
        epochs=epochs,
    )
    analytic_current = hilbert_transform(current_signal)
    amplitude_restored = np.abs(analytic_current)
    phase_current = np.unwrap(np.angle(analytic_current))
    frequency_restored = np.diff(phase_current) / (2 * np.pi) * fs

    return {
        "analytic_noisy": analytic_observed,
        "amplitude_noisy": amplitude_noisy,
        "frequency_noisy": frequency_noisy,
        "amplitude_restored": amplitude_restored,
        "frequency_restored": frequency_restored,
        "chirp_denoised": current_signal,
    }


def extract_clean_features(chirp_clean, fs):
    analytic_clean = hilbert_transform(chirp_clean)
    amplitude_clean = np.abs(analytic_clean)
    phase_clean = np.unwrap(np.angle(analytic_clean))
    frequency_clean = np.diff(phase_clean) / (2 * np.pi) * fs
    return {
        "amplitude_clean": amplitude_clean,
        "frequency_clean": frequency_clean,
    }


def load_dcnn(device, signal_length, output_scale):
    model = DCNN(input_channels=1, N=signal_length, output_scale=output_scale)
    model.load_state_dict(torch.load(DCNN_DIR / "dcnn_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def dcnn_denoise(model, noisy_signal, normalize_scale, device):
    with torch.no_grad():
        noisy_tensor = torch.FloatTensor(noisy_signal / normalize_scale).unsqueeze(0).unsqueeze(0).to(device)
        denoised_tensor = model(noisy_tensor)
    return denoised_tensor.squeeze().cpu().numpy() * normalize_scale


def load_transformer(device):
    model = Transformer(
        input_dim=1,
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_encoder_blocks=6,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(TRANSFORMER_DIR / "transformer_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def transformer_denoise(model, noisy_signal, device):
    with torch.no_grad():
        noisy_tensor = (
            torch.FloatTensor(noisy_signal / NORMALIZATION_SCALE).unsqueeze(0).unsqueeze(-1).to(device)
        )
        denoised_tensor = model(src=noisy_tensor)
    return denoised_tensor.squeeze().cpu().numpy() * NORMALIZATION_SCALE


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
    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.08, 1)

    boxes = [
        (0.04, 0.34, 0.13, 0.32, "Noisy chirp\nsignal"),
        (0.22, 0.34, 0.14, 0.32, "Feature\nextraction"),
        (0.41, 0.58, 0.14, 0.16, "Amplitude\nfeature"),
        (0.41, 0.26, 0.14, 0.16, "Frequency\nfeature"),
        (0.60, 0.58, 0.14, 0.16, "GGLR\n denoising"),
        (0.60, 0.26, 0.14, 0.16, "GGLR\n denoising"),
        (0.79, 0.34, 0.12, 0.32, "Signal\nintegration"),
        (0.97, 0.34, 0.07, 0.32, "Iterative\nupdate"),
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
        ((0.17, 0.50), (0.22, 0.50)),
        ((0.36, 0.56), (0.41, 0.66)),
        ((0.36, 0.44), (0.41, 0.34)),
        ((0.55, 0.66), (0.60, 0.66)),
        ((0.55, 0.34), (0.60, 0.34)),
        ((0.74, 0.66), (0.79, 0.58)),
        ((0.74, 0.34), (0.79, 0.42)),
        ((0.91, 0.50), (0.97, 0.50)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.8,
                color="#274c77",
                shrinkA=2,
                shrinkB=2,
            )
        )

    ax.add_patch(
        FancyArrowPatch(
            (1.005, 0.34),
            (0.29, 0.34),
            connectionstyle="arc3,rad=-0.32",
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#274c77",
            shrinkA=2,
            shrinkB=2,
            clip_on=False,
        )
    )
    ax.set_title("Overall framework of the iterative chirp denoising method.")
    save_figure(fig, "fig1_framework.png")


def plot_observation_and_restoration_figures(t, clean_features, proposed_features):
    t_freq = t[: len(clean_features["frequency_clean"])]
    t_amp = t[: len(clean_features["amplitude_clean"])]

    frequency_clean = clean_features["frequency_clean"]
    amplitude_clean = clean_features["amplitude_clean"]
    frequency_noisy = proposed_features["frequency_noisy"][: len(clean_features["frequency_clean"])]
    amplitude_noisy = proposed_features["amplitude_noisy"][: len(clean_features["amplitude_clean"])]
    frequency_restored = proposed_features["frequency_restored"][: len(clean_features["frequency_clean"])]
    amplitude_restored = proposed_features["amplitude_restored"][: len(clean_features["amplitude_clean"])]

    styles = {
        "observed": {"color": "#e76f51", "linewidth": 1.0, "linestyle": "-", "alpha": 0.9},
        "restored": {"color": "#1d3557", "linewidth": 1.8, "linestyle": "-.", "alpha": 0.95},
        "theoretical": {"color": "#2a9d8f", "linewidth": 1.5, "linestyle": "--", "alpha": 0.95},
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_freq, frequency_noisy, label="Observed frequency", **styles["observed"])
    ax.plot(t_freq, frequency_clean, label="Theoretical frequency", **styles["theoretical"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Observed and theoretical instantaneous frequency.")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig2_hilbert_frequency_observation.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_amp, amplitude_noisy, label="Observed amplitude", **styles["observed"])
    ax.plot(t_amp, amplitude_clean, label="Theoretical amplitude", **styles["theoretical"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Observed and theoretical amplitude envelope.")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig3_hilbert_amplitude_observation.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_amp, amplitude_noisy, label="Observed amplitude", **styles["observed"])
    ax.plot(t_amp, amplitude_restored, label="Restored amplitude", **styles["restored"])
    ax.plot(t_amp, amplitude_clean, label="Theoretical amplitude", **styles["theoretical"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Amplitude restoration after the proposed denoising.")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig4_gglr_amplitude_restoration.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_freq, frequency_noisy, label="Observed frequency", **styles["observed"])
    ax.plot(t_freq, frequency_restored, label="Restored frequency", **styles["restored"])
    ax.plot(t_freq, frequency_clean, label="Theoretical frequency", **styles["theoretical"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Instantaneous frequency restoration after the proposed denoising.")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig5_gglr_frequency_restoration.png")


def plot_noisy_signal_vs_clean(t, clean_signal, noisy_signal):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, clean_signal, color="#2a9d8f", linewidth=1.4, label="Clean signal")
    ax.plot(t, noisy_signal, color="#e76f51", linewidth=1.0, linestyle="--", alpha=0.85, label="Noisy signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Clean and noisy chirp signals.")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "fig4_noisy_signal_vs_clean.png")


def plot_local_method_comparison(t, clean_signal, noisy_signal, denoised_signals, window_size=300):
    center = len(t) // 2
    start = max(0, center - window_size // 2)
    end = start + window_size

    t_local = t[start:end]
    clean_local = clean_signal[start:end]
    noisy_local = noisy_signal[start:end]
    x_limits = (t_local[0], t_local[-1])

    colors = {
        "GGLR": "#1d3557",
        "DCNN": "#6a994e",
        "Transformer-Encoder": "#7b2cbf",
    }

    method_items = list(denoised_signals.items())
    denoised_locals = {method_name: signal[start:end] for method_name, signal in method_items}
    error_locals = {method_name: denoised_local - clean_local for method_name, denoised_local in denoised_locals.items()}

    signal_stack = np.concatenate([clean_local, noisy_local, *denoised_locals.values()])
    signal_margin = 0.08 * np.ptp(signal_stack) if np.ptp(signal_stack) > 0 else 1.0
    signal_limits = (signal_stack.min() - signal_margin, signal_stack.max() + signal_margin)

    error_stack = np.concatenate(list(error_locals.values()))
    error_margin = 0.1 * np.ptp(error_stack) if np.ptp(error_stack) > 0 else 0.1
    error_peak = max(abs(error_stack.min()), abs(error_stack.max())) + error_margin
    error_limits = (-error_peak, error_peak)

    fig = plt.figure(figsize=(15.5, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.15, 1.0, 1.0, 1.0])

    noisy_ax = fig.add_subplot(grid[:, 0])
    noisy_ax.plot(t_local, clean_local, color="#2a9d8f", linewidth=1.2, linestyle="--", label="Clean")
    noisy_ax.plot(t_local, noisy_local, color="#e76f51", linewidth=1.0, alpha=0.9, label="Noisy")
    noisy_ax.set_title("(a) Local noisy signal")
    noisy_ax.set_xlabel("Time (s)")
    noisy_ax.set_ylabel("Amplitude")
    noisy_ax.set_xlim(*x_limits)
    noisy_ax.set_ylim(*signal_limits)
    noisy_ax.grid(True, alpha=0.3)
    noisy_ax.legend(loc="upper right", fontsize=9)

    for idx, (method_name, denoised_signal) in enumerate(method_items):
        denoised_local = denoised_locals[method_name]
        error_local = error_locals[method_name]
        signal_ax = fig.add_subplot(grid[0, idx + 1], sharex=noisy_ax)
        error_ax = fig.add_subplot(grid[1, idx + 1], sharex=noisy_ax)

        signal_ax.plot(t_local, clean_local, color="#2a9d8f", linewidth=1.2, linestyle="--", label="Clean")
        signal_ax.plot(t_local, denoised_local, color=colors[method_name], linewidth=1.0, label=method_name)
        signal_ax.set_title(f"({chr(ord('b') + idx)}) {method_name} restoration")
        signal_ax.set_xlim(*x_limits)
        signal_ax.set_ylim(*signal_limits)
        signal_ax.grid(True, alpha=0.3)
        signal_ax.legend(loc="upper right", fontsize=9)
        if idx == 0:
            signal_ax.set_ylabel("Amplitude")

        error_ax.plot(t_local, error_local, color=colors[method_name], linewidth=1.0)
        error_ax.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.7)
        error_ax.set_title(f"({chr(ord('e') + idx)}) {method_name} error")
        error_ax.set_xlabel("Time (s)")
        error_ax.set_xlim(*x_limits)
        error_ax.set_ylim(*error_limits)
        error_ax.grid(True, alpha=0.3)
        if idx == 0:
            error_ax.set_ylabel("Error")

    fig.suptitle(f"Local comparison of denoising results over the middle {window_size} samples.")
    save_figure(fig, "fig5_local_methods_comparison.png")


def plot_quantitative_figures(results):
    sigma_values = [row["sigma"] for row in results["GGLR"]]
    method_colors = {
        "GGLR": "#1d3557",
        "DCNN": "#6a994e",
        "Transformer-Encoder": "#7b2cbf",
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
            sigmaa=600,
            lambdaa=100,
            sigmaf=600,
            lambdaf=lambda_reg,
            epochs=3,
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


def export_comparison_table(results, a1_value, epochs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = OUTPUT_DIR / "comparison_a1_20_epoch3.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["a1", "epochs", "method", "sigma", "snr_input", "snr_output", "snr_gain"])
        for method_name, rows in results.items():
            for row in rows:
                writer.writerow(
                    [a1_value, epochs, method_name, row["sigma"], row["snr_input"], row["snr_output"], row["snr_gain"]]
                )


def main():
    setup_style()

    fs = 1000
    T = 1.0
    f0 = 50.0
    f1 = 200.0
    a0 = 50.0
    a1 = 20.0
    sigma_focus = 10.0
    sigma_list = [20, 10, 5, 2, 1, 0.5]
    seed = 42
    ignore_ratio = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    normalize_scale = a0
    dcnn_output_scale = 1.0
    sigmaa = 600
    lambdaa = 100
    sigmaf = 600
    lambdaf = 0.1
    epochs = 3

    plot_framework()

    t, chirp_clean, chirp_noisy = generate_chirp_signal(
        fs, T, f0, f1, a0=a0, a1=a1, sigma=sigma_focus, seed=seed
    )
    clean_features = extract_clean_features(chirp_clean, fs)
    proposed_features = proposed_denoise(
        chirp_noisy, fs, sigmaa=sigmaa, lambdaa=lambdaa, sigmaf=sigmaf, lambdaf=lambdaf, epochs=epochs
    )
    chirp_proposed = proposed_features["chirp_denoised"]

    dcnn_model = load_dcnn(device, len(chirp_clean), output_scale=dcnn_output_scale)
    transformer_model = load_transformer(device)
    chirp_dcnn = dcnn_denoise(dcnn_model, chirp_noisy, normalize_scale=normalize_scale, device=device)
    chirp_transformer = transformer_denoise(transformer_model, chirp_noisy, device)

    plot_observation_and_restoration_figures(t, clean_features, proposed_features)
    plot_noisy_signal_vs_clean(t, chirp_clean, chirp_noisy)
    plot_local_method_comparison(
        t,
        chirp_clean,
        chirp_noisy,
        {
            "GGLR": chirp_proposed,
            "DCNN": chirp_dcnn,
            "Transformer-Encoder": chirp_transformer,
        },
        window_size=300,
    )

    results = {"GGLR": [], "DCNN": [], "Transformer-Encoder": []}
    for sigma_noise in sigma_list:
        _, chirp_clean_sigma, chirp_noisy_sigma = generate_chirp_signal(
            fs, T, f0, f1, a0=a0, a1=a1, sigma=sigma_noise, seed=seed
        )
        chirp_proposed_sigma = proposed_denoise(
            chirp_noisy_sigma,
            fs,
            sigmaa=sigmaa,
            lambdaa=lambdaa,
            sigmaf=sigmaf,
            lambdaf=lambdaf,
            epochs=epochs,
        )["chirp_denoised"]
        chirp_dcnn_sigma = dcnn_denoise(
            dcnn_model,
            chirp_noisy_sigma,
            normalize_scale=normalize_scale,
            device=device,
        )
        chirp_transformer_sigma = transformer_denoise(transformer_model, chirp_noisy_sigma, device)

        snr_input = calculate_snr(chirp_clean_sigma, chirp_noisy_sigma, ignore_ratio=ignore_ratio)
        method_outputs = {
            "GGLR": chirp_proposed_sigma,
            "DCNN": chirp_dcnn_sigma,
            "Transformer-Encoder": chirp_transformer_sigma,
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
    export_comparison_table(results, a1, epochs)

    print(f"All figures have been saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
