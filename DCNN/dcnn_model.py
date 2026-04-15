import torch
import torch.nn as nn

class DCNN_VGG(nn.Module):
    def __init__(self, input_channels, num_channels, num_layers, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = num_channels
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                nn.Conv1d(
                    in_channels=self.input_channels,
                    out_channels=self.output_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1
                )
            )
            self.layers.append(
                nn.BatchNorm1d(num_features=self.output_channels)
            )
            self.layers.append(
                nn.Tanh()
            )
            self.input_channels = self.output_channels
        self.net = nn.Sequential(*self.layers)

    def forward(self, X):
        return self.net(X)


class DCNN(nn.Module):
    def __init__(self, input_channels, N, **kwargs):
        super(DCNN, self).__init__(**kwargs)
        
        self.vgg1 = DCNN_VGG(
            input_channels=input_channels,
            num_channels=64,
            num_layers=10,
            kernel_size=3
        )
        self.vgg2 = DCNN_VGG(
            input_channels=64,
            num_channels=32,
            num_layers=10,
            kernel_size=5
        )
        self.vgg3 = DCNN_VGG(
            input_channels=32,
            num_channels=16,
            num_layers=10,
            kernel_size=7
        )
        self.vgg4 = DCNN_VGG(
            input_channels=16,
            num_channels=8,
            num_layers=1,
            kernel_size=9
        )
        
        # 全连接层
        self.flatten = nn.Flatten()               # (batch, 8, N) -> (batch, 8*N)
        self.dense = nn.Linear(8 * N, N)          # (batch, 8*N) -> (batch, N)
        self.output_activation = nn.Tanh()

    def forward(self, X):
        X = self.vgg1(X)      # (batch, 64, N)
        X = self.vgg2(X)      # (batch, 32, N)
        X = self.vgg3(X)      # (batch, 16, N)
        X = self.vgg4(X)      # (batch, 8, N)
        X = self.flatten(X)   # (batch, 8*N)
        X = self.dense(X)     # (batch, N)
        X = X.unsqueeze(1)    # (batch, 1, N)  
        X = self.output_activation(X)
        return X