import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dcnn_data import ChirpDataset
from dcnn_model import DCNN


def train(model, dataloader, epochs=100, lr=1e-3, device="cuda", save_path="dcnn_model.pth"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        for noisy, clean in dataloader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model. Loss: {best_loss:.6f}")

    print(f"Training finished. Best loss: {best_loss:.6f}")
    return model


if __name__ == "__main__":
    fs = 1000
    T = 1
    N = int(fs * T)
    a0 = 50.0
    end_amplitudes = [40.0, 20.0, 1.0]
    num_samples = 5000
    batch_size = 32
    epochs = 100
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Amplitude decay modes: a0={a0}, a1 in {end_amplitudes}")

    dataset = ChirpDataset(
        num_samples=num_samples,
        fs=fs,
        T=T,
        seed=42,
        a0=a0,
        end_amplitudes=end_amplitudes,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DCNN(input_channels=1, N=N)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")

    model = train(model, dataloader, epochs=epochs, lr=lr, device=device, save_path="dcnn_model.pth")
