import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim


class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 16)
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 28 * 28), nn.Sigmoid()  # we want outputs between 0 - 1
        )

    def forward(self, x):
        low_dimensional_space = self.encoder(x)
        reconstructed_image = self.decoder(low_dimensional_space)

        return reconstructed_image
