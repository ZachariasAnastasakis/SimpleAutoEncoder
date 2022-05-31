from Autoencoder import Autoencoder
from prepare_dataset import LoadDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# TODO: add tensorboards
model = torch.load('./model.pth')
model.eval()

data = LoadDataset(dataset="MNIST", mode="test")
data_loader = data.DataLoader()

imgs = []
recon = []
for i, (image, label) in enumerate(data_loader):
    image = image.reshape(-1, 28 * 28)
    imgs.append(image.detach().numpy())
    res = model(image)
    recon.append(res.detach().numpy())

    if i == 8: break

plt.figure(figsize=(9, 3))
plt.gray()

for i, item in enumerate(imgs):
    if i >= 9: break
    if i == 0: plt.subplot(2, 9, i + 1, title="Original Images")
    else: plt.subplot(2, 9, i + 1)
    item = item.reshape(-1, 28, 28)
    # item: 1, 28, 28
    plt.axis('off')
    plt.imshow(item[0])

for i, item in enumerate(recon):
    if i >= 9: break
    if i == 0: plt.subplot(2, 9, 9 + i + 1, title="Reconstructed Images")
    else: plt.subplot(2, 9, 9 + i + 1)
    item = item.reshape(-1, 28, 28)
    # item: 1, 28, 28
    plt.axis('off')
    plt.imshow(item[0])

plt.show()
