import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from transformer_data import ChirpDataset
from transformer_model import Transformer

def train(model, dataloader, epochs=100, lr=1e-3, device='cuda', save_path='transformer_model.pth'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        for noisy, clean in dataloader:
            noisy = noisy.to(device)  # (batch, N)
            clean = clean.to(device)  # (batch, N)
            
            # Transformer编码器输入需要 (batch, seq_len, features)
            noisy = noisy.unsqueeze(-1)  # (batch, N, 1)
            clean_target = clean.unsqueeze(-1)  # (batch, N, 1)
            
            optimizer.zero_grad()
            output = model(src=noisy)  # (batch, N, 1)
            loss = criterion(output, clean_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型, Loss: {best_loss:.6f}")
    
    print(f"训练完成，最佳损失: {best_loss:.6f}")
    return model

if __name__ == "__main__":
    # 参数设置
    fs = 1000
    T = 1
    N = int(fs * T)
    num_samples = 5000
    batch_size = 32
    epochs = 100
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Transformer参数
    input_dim = 1
    embed_dim = 64
    num_heads = 4
    ff_dim = 128
    num_encoder_blocks = 6
    dropout = 0.1
    
    print(f"使用设备: {device}")
    print(f"Transformer参数: embed_dim={embed_dim}, num_heads={num_heads}, num_encoder_blocks={num_encoder_blocks}")
    
    # 加载数据
    dataset = ChirpDataset(num_samples=num_samples, fs=fs, T=T, seed=42)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型（编码器only）
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
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    model = train(model, dataloader, epochs=epochs, lr=lr, device=device, save_path='transformer_model.pth')