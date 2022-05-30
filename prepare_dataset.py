import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.optim as optim


class LoadDataset:

    def __init__(self, dataset="MNIST", batch_size=64, mode="train"):
        if mode == "train":
            self.mode = True
        else:
            self.mode = False
        self.batch_size = batch_size
        transform = transforms.ToTensor()
        if dataset == "MNIST":
            self.data = datasets.MNIST(root='./data', train=self.mode, download=True, transform=transform)

    def DataLoader(self):
        train_data_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=True)

        return train_data_loader

# images, labels = next(iter(data_loader))
# plt.imshow(images[0].reshape(28, 28), cmap="gray")
# plt.show()
