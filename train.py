import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from prepare_dataset import LoadDataset
from Autoencoder import Autoencoder
# add tensor boards
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = LoadDataset(dataset="MNIST")
data_loader = data.DataLoader()

autoencoder = Autoencoder()
epochs = 20
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(epochs):
    for image, label in tqdm(data_loader, desc=f'Epoch {epoch}/{epochs}'):
        image = image.reshape(-1, 28 * 28)
        reconstructed_img = autoencoder(image)
        loss = criterion(reconstructed_img, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"Training finished with loss: {loss.item()}")
torch.save(autoencoder, './model.pth')
