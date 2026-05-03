import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ChirpDataset(Dataset):
    def __init__(
        self,
        num_samples=5000,
        fs=1000,
        T=1,
        seed=42,
        a0=50.0,
        end_amplitudes=(50.0, 40.0, 20.0, 1.0),
        normalize_scale=None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.fs = fs
        self.T = T
        self.N = int(fs * T)
        self.a0 = float(a0)
        self.end_amplitudes = np.asarray(end_amplitudes, dtype=np.float32)
        self.normalize_scale = float(normalize_scale) if normalize_scale is not None else float(a0)

        if self.end_amplitudes.ndim != 1 or len(self.end_amplitudes) == 0:
            raise ValueError("end_amplitudes must be a non-empty 1D sequence.")
        if self.normalize_scale <= 0:
            raise ValueError("normalize_scale must be positive.")

        np.random.seed(seed)

        self.f0_list = np.random.uniform(45, 55, num_samples)
        self.f1_list = np.random.uniform(195, 205, num_samples)
        self.sigma_list = np.random.uniform(0.5, 20, num_samples)
        self.phis_list = np.random.uniform(0, 2 * np.pi, num_samples)
        self.a1_list = np.random.choice(self.end_amplitudes, size=num_samples)

        self.clean_signals = []
        self.noisy_signals = []

        for i in range(num_samples):
            t = np.linspace(0, T, self.N, endpoint=False)
            k = (self.f1_list[i] - self.f0_list[i]) / T
            ka = (self.a1_list[i] - self.a0) / T
            envelope = self.a0 + ka * t
            chirp_clean = envelope * np.cos(
                2 * np.pi * (self.f0_list[i] * t + 0.5 * k * t**2) + self.phis_list[i]
            )

            noise = self.sigma_list[i] * np.random.randn(self.N)
            chirp_noisy = chirp_clean + noise

            self.clean_signals.append(chirp_clean)
            self.noisy_signals.append(chirp_noisy)

        self.clean_signals = np.array(self.clean_signals, dtype=np.float32)
        self.noisy_signals = np.array(self.noisy_signals, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        clean = torch.FloatTensor(self.clean_signals[idx] / self.normalize_scale).unsqueeze(0)
        noisy = torch.FloatTensor(self.noisy_signals[idx] / self.normalize_scale).unsqueeze(0)
        return noisy, clean


def generate_data(
    num_samples=5000,
    fs=1000,
    T=1,
    batch_size=32,
    shuffle=True,
    seed=42,
    a0=50.0,
    end_amplitudes=(50.0, 40.0, 20.0, 1.0),
    normalize_scale=None,
):
    dataset = ChirpDataset(
        num_samples=num_samples,
        fs=fs,
        T=T,
        seed=seed,
        a0=a0,
        end_amplitudes=end_amplitudes,
        normalize_scale=normalize_scale,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset


if __name__ == "__main__":
    dataloader, dataset = generate_data(num_samples=5000)
    print(f"Dataset size: {len(dataset)}")
    print(f"Amplitude decay modes: {dataset.end_amplitudes.tolist()}")
    for noisy, clean in dataloader:
        print(f"batch - noisy shape: {noisy.shape}, clean shape: {clean.shape}")
        break
