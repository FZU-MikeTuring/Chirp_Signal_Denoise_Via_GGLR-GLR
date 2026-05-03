import csv

import matplotlib.pylab as plt
import numpy as np

try:
    from scipy.signal import hilbert
except ImportError:
    def hilbert(signal):
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


def generate_chirp_signal(fs=1000, T=1, f0=50, f1=200, a0=50, a1=1, sigma=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, T, int(fs * T), endpoint=False)
    k = (f1 - f0) / T
    ka = (a1 - a0) / T
    chirp_clean = (a0 + ka * t) * np.cos(2 * np.pi * (f0 * t + 0.5 * k * t**2) + np.pi)
    chirp_noisy = chirp_clean + sigma * np.random.randn(len(t))
    return chirp_clean, chirp_noisy


def calculate_snr(clean_signal, noisy_signal, ignore_ratio=0.1):
    n = len(clean_signal)
    ignore = int(n * ignore_ratio)
    clean_mid = clean_signal[ignore:-ignore]
    noisy_mid = noisy_signal[ignore:-ignore]
    noise = noisy_mid - clean_mid
    signal_power = np.mean(clean_mid**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)


def GGLR(signal_values, sigma=50, lambda_reg=0.1):
    m = len(signal_values)
    grad_m = m - 1
    k_neighbors = grad_m // 2
    grad_operator = np.zeros((grad_m, m), dtype=float)
    for i in range(grad_m):
        grad_operator[i, i] = -1
        grad_operator[i, i + 1] = 1

    grad_values = signal_values[1:] - signal_values[:-1]
    weights = np.zeros((grad_m, grad_m), dtype=float)

    for i in range(grad_m):
        distances = np.abs(grad_values - grad_values[i]).flatten()
        neighbors = np.argsort(distances)[1 : k_neighbors + 1]
        for j in neighbors:
            weight = np.exp(-distances[j] ** 2 / (2 * sigma**2))
            weights[i, j] = weight
            weights[j, i] = weight

    degree = np.diag(weights.sum(axis=1))
    laplacian = degree - weights
    smoothness = grad_operator.T @ laplacian @ grad_operator
    system = np.eye(m) + lambda_reg * smoothness
    return np.linalg.solve(system, signal_values)


def denoise(chirp_noisy, fs, sigmaa, lambdaa, sigmaf, lambdaf, epochs=1):
    current_signal = chirp_noisy.copy()

    for _ in range(epochs):
        analytic_noisy = hilbert(current_signal)
        amplitude_noisy = np.abs(analytic_noisy)
        phase_noisy = np.unwrap(np.angle(analytic_noisy))
        init_phase = phase_noisy[0]
        frequency_noisy = np.diff(phase_noisy) / (2 * np.pi) * fs

        amplitude_denoised = GGLR(amplitude_noisy, sigmaa, lambdaa)
        frequency_denoised = GGLR(frequency_noisy, sigmaf, lambdaf)

        phase_increments = np.cumsum(frequency_denoised) * (2 * np.pi / fs)
        phase_reconstructed = np.full(len(phase_increments) + 1, init_phase)
        phase_reconstructed[1:] += phase_increments
        current_signal = np.real(amplitude_denoised * np.exp(1j * phase_reconstructed))

    return current_signal


def export_metrics(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "sigma", "a1", "decay_ratio", "snr_input", "snr_output", "snr_gain"])
        for row in results:
            writer.writerow(
                [
                    row.get("epoch"),
                    row.get("sigma"),
                    row["a1"],
                    row["decay_ratio"],
                    row["snr_input"],
                    row["snr_output"],
                    row["snr_gain"],
                ]
            )


def plot_heatmap(ax, matrix, x_labels, y_labels, xlabel, ylabel, title, colorbar_label):
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="upper")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([str(x) for x in x_labels])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([str(y) for y in y_labels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)

    color_bar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    color_bar.set_label(colorbar_label)


def plot_sigma_terminal_heatmaps(results, sigma_list, end_amplitudes):
    output_matrix = np.zeros((len(sigma_list), len(end_amplitudes)))
    gain_matrix = np.zeros((len(sigma_list), len(end_amplitudes)))

    for i, sigma_noise in enumerate(sigma_list):
        for j, a1_value in enumerate(end_amplitudes):
            matched_row = next(
                row
                for row in results
                if row["sigma"] == sigma_noise and row["a1"] == a1_value and row.get("epoch") is None
            )
            output_matrix[i, j] = matched_row["snr_output"]
            gain_matrix[i, j] = matched_row["snr_gain"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    plot_heatmap(
        axes[0],
        output_matrix,
        end_amplitudes,
        sigma_list,
        "Terminal amplitude",
        "Noise standard deviation $\\sigma$",
        "Output SNR under different noise levels and terminal amplitudes",
        "Output SNR (dB)",
    )
    plot_heatmap(
        axes[1],
        gain_matrix,
        end_amplitudes,
        sigma_list,
        "Terminal amplitude",
        "Noise standard deviation $\\sigma$",
        "SNR gain under different noise levels and terminal amplitudes",
        "SNR gain (dB)",
    )
    plt.tight_layout()
    plt.savefig("sigma_terminal_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_epoch_terminal_heatmaps(results, epoch_list, end_amplitudes, sigma_noise):
    output_matrix = np.zeros((len(epoch_list), len(end_amplitudes)))
    gain_matrix = np.zeros((len(epoch_list), len(end_amplitudes)))

    for i, epoch in enumerate(epoch_list):
        for j, a1_value in enumerate(end_amplitudes):
            matched_row = next(
                row for row in results if row["epoch"] == epoch and row["a1"] == a1_value
            )
            output_matrix[i, j] = matched_row["snr_output"]
            gain_matrix[i, j] = matched_row["snr_gain"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    plot_heatmap(
        axes[0],
        output_matrix,
        end_amplitudes,
        epoch_list,
        "Terminal amplitude",
        "Epochs",
        f"Output SNR under fixed noise standard deviation $\\sigma$={sigma_noise}",
        "Output SNR (dB)",
    )
    plot_heatmap(
        axes[1],
        gain_matrix,
        end_amplitudes,
        epoch_list,
        "Terminal amplitude",
        "Epochs",
        f"SNR gain under fixed noise standard deviation $\\sigma$={sigma_noise}",
        "SNR gain (dB)",
    )
    plt.tight_layout()
    plt.savefig("epoch_terminal_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_sigma_terminal_experiment(
    fs,
    T,
    f0,
    f1,
    a0,
    end_amplitudes,
    sigma_list,
    random_seed,
    ignore_ratio,
    sigmaa,
    lambdaa,
    sigmaf,
    lambdaf,
    epochs,
):
    results = []
    for sigma_noise in sigma_list:
        for a1_value in end_amplitudes:
            chirp_clean, chirp_noisy = generate_chirp_signal(
                fs, T, f0, f1, a0, a1_value, sigma=sigma_noise, seed=random_seed
            )
            chirp_denoised = denoise(chirp_noisy, fs, sigmaa, lambdaa, sigmaf, lambdaf, epochs=epochs)
            snr_input = calculate_snr(chirp_clean, chirp_noisy, ignore_ratio)
            snr_output = calculate_snr(chirp_clean, chirp_denoised, ignore_ratio)
            results.append(
                {
                    "epoch": None,
                    "sigma": sigma_noise,
                    "a1": a1_value,
                    "decay_ratio": (a0 - a1_value) / a0,
                    "snr_input": snr_input,
                    "snr_output": snr_output,
                    "snr_gain": snr_output - snr_input,
                }
            )

    export_metrics(results, "sigma_terminal_heatmap_metrics.csv")
    plot_sigma_terminal_heatmaps(results, sigma_list, end_amplitudes)
    return results


def run_epoch_terminal_experiment(
    fs,
    T,
    f0,
    f1,
    a0,
    end_amplitudes,
    sigma_noise,
    epoch_list,
    random_seed,
    ignore_ratio,
    sigmaa,
    lambdaa,
    sigmaf,
    lambdaf,
):
    results = []
    epoch_list = sorted(set(epoch_list))

    for a1_value in end_amplitudes:
        chirp_clean, chirp_noisy = generate_chirp_signal(
            fs, T, f0, f1, a0, a1_value, sigma=sigma_noise, seed=random_seed
        )
        snr_input = calculate_snr(chirp_clean, chirp_noisy, ignore_ratio)

        for epoch in epoch_list:
            chirp_denoised = denoise(chirp_noisy, fs, sigmaa, lambdaa, sigmaf, lambdaf, epochs=epoch)
            snr_output = calculate_snr(chirp_clean, chirp_denoised, ignore_ratio)
            results.append(
                {
                    "epoch": epoch,
                    "sigma": sigma_noise,
                    "a1": a1_value,
                    "decay_ratio": (a0 - a1_value) / a0,
                    "snr_input": snr_input,
                    "snr_output": snr_output,
                    "snr_gain": snr_output - snr_input,
                }
            )

    export_metrics(results, "epoch_terminal_heatmap_metrics.csv")
    plot_epoch_terminal_heatmaps(results, epoch_list, end_amplitudes, sigma_noise)
    return results


def main():
    fs = 1000
    T = 1
    f0 = 50
    f1 = 200
    a0 = 50
    end_amplitudes = [50, 40, 20, 1]

    ignore_ratio = 0.1
    random_seed = 42

    sigmaf = 600
    lambdaf = 0.1
    sigmaa = 600
    lambdaa = 100

    sigma_list = [20, 10, 5, 2, 1, 0.5]
    epochs = 1

    epoch_list = [1, 2, 3, 5, 7, 9, 10]
    epoch_terminal_sigma = 10

    np.random.seed(random_seed)

    run_sigma_terminal_experiment(
        fs=fs,
        T=T,
        f0=f0,
        f1=f1,
        a0=a0,
        end_amplitudes=end_amplitudes,
        sigma_list=sigma_list,
        random_seed=random_seed,
        ignore_ratio=ignore_ratio,
        sigmaa=sigmaa,
        lambdaa=lambdaa,
        sigmaf=sigmaf,
        lambdaf=lambdaf,
        epochs=epochs,
    )

    run_epoch_terminal_experiment(
        fs=fs,
        T=T,
        f0=f0,
        f1=f1,
        a0=a0,
        end_amplitudes=end_amplitudes,
        sigma_noise=epoch_terminal_sigma,
        epoch_list=epoch_list,
        random_seed=random_seed,
        ignore_ratio=ignore_ratio,
        sigmaa=sigmaa,
        lambdaa=lambdaa,
        sigmaf=sigmaf,
        lambdaf=lambdaf,
    )


if __name__ == "__main__":
    main()
