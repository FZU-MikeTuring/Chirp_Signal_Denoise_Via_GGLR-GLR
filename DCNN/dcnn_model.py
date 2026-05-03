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
                    stride=1,
                )
            )
            self.layers.append(nn.BatchNorm1d(num_features=self.output_channels))
            self.layers.append(nn.Tanh())
            self.input_channels = self.output_channels
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)


class DCNN(nn.Module):
    def __init__(self, input_channels, N, output_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.output_scale = float(output_scale)

        self.vgg1 = DCNN_VGG(
            input_channels=input_channels,
            num_channels=64,
            num_layers=10,
            kernel_size=3,
        )
        self.vgg2 = DCNN_VGG(
            input_channels=64,
            num_channels=32,
            num_layers=10,
            kernel_size=5,
        )
        self.vgg3 = DCNN_VGG(
            input_channels=32,
            num_channels=16,
            num_layers=10,
            kernel_size=7,
        )
        self.vgg4 = DCNN_VGG(
            input_channels=16,
            num_channels=8,
            num_layers=1,
            kernel_size=9,
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8 * N, N)
        self.output_activation = nn.Tanh()

    def forward(self, x):
        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.vgg3(x)
        x = self.vgg4(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = x.unsqueeze(1)
        x = self.output_activation(x) * self.output_scale
        return x
