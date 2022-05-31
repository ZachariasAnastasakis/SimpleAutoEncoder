import torch
import torch.nn as nn

from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset import LoadDataset
from Autoencoder import Autoencoder

writer = SummaryWriter('runs/AutoEncoder_train_loss')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = LoadDataset(dataset="MNIST")
data_loader = data.DataLoader()
total_steps = len(data_loader)

autoencoder = Autoencoder()
epochs = 20
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)

running_loss = 0
i = 0
for epoch in range(epochs):
    for image, label in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
        image = image.reshape(-1, 28 * 28)
        reconstructed_img = autoencoder(image)
        loss = criterion(reconstructed_img, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/Train', loss.item(), epoch * total_steps + 1)

print(f"Training finished with loss: {loss.item()}")
torch.save(autoencoder, './model.pth')
