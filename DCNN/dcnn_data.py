import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class ChirpDataset(Dataset):
    def __init__(self, num_samples=5000, fs=1000, T=1, seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.fs = fs
        self.T = T
        self.N = int(fs * T)
        
        np.random.seed(seed)
        
        # 随机生成参数
        self.f0_list = np.random.uniform(45, 55, num_samples)
        self.f1_list = np.random.uniform(195, 205, num_samples)
        self.sigma_list = np.random.uniform(0.02, 2.5, num_samples)
        self.phis_list = np.random.uniform(0, 2 * np.pi, num_samples)
        
        # 生成干净信号和带噪信号
        self.clean_signals = []
        self.noisy_signals = []
        
        for i in range(num_samples):
            t = np.linspace(0, T, self.N, endpoint=False)
            k = (self.f1_list[i] - self.f0_list[i]) / T
            chirp_clean = np.cos(2 * np.pi * (self.f0_list[i] * t + 0.5 * k * t**2) + self.phis_list[i])
            
            noise = self.sigma_list[i] * np.random.randn(self.N)
            chirp_noisy = chirp_clean + noise
            
            self.clean_signals.append(chirp_clean)
            self.noisy_signals.append(chirp_noisy)
        
        self.clean_signals = np.array(self.clean_signals)
        self.noisy_signals = np.array(self.noisy_signals)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        clean = torch.FloatTensor(self.clean_signals[idx]).unsqueeze(0)  # (1, N)
        noisy = torch.FloatTensor(self.noisy_signals[idx]).unsqueeze(0)  # (1, N)
        return noisy, clean

def generate_data(num_samples=5000, fs=1000, T=1, batch_size=32, shuffle=True, seed=42):
    dataset = ChirpDataset(num_samples, fs, T, seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

if __name__ == "__main__":
    dataloader, dataset = generate_data(num_samples=5000)
    print(f"数据集大小: {len(dataset)}")
    for noisy, clean in dataloader:
        print(f"batch - noisy shape: {noisy.shape}, clean shape: {clean.shape}")
        break