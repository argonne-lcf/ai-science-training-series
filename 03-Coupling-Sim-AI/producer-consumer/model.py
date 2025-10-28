import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """Simple autoregressive CNN model to predict the next step of a wave equation
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        # Encoder: extract spatial features
        self.conv1 = nn.Conv2d(1, 8, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, padding='same')
        # Decoder: reconstruct same-sized output
        self.conv3 = nn.Conv2d(16, 8, kernel_size=kernel_size, padding='same')
        self.conv4 = nn.Conv2d(8, 1, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)   # last layer linear (no activation) to produce real values
        return x