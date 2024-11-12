import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, 20)  # Mean
        self.fc22 = nn.Linear(400, 20)  # Log variance
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

vae = VAE().to(device)
vae.load_state_dict(torch.load('vae_model.pth'))  # Load the trained model
vae.eval()

def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clip(noisy_images, 0., 1.)
    return noisy_images

original_images, _ = next(iter(train_loader))
noisy_images = add_noise(original_images.to(device))

with torch.no_grad():
    reconstructions, _, _ = vae(noisy_images.view(-1, 28 * 28))

reconstruction_errors = torch.mean((noisy_images.view(-1, 28 * 28) - reconstructions) ** 2, dim=1)

plt.hist(reconstruction_errors.cpu().numpy(), bins=50, alpha=0.7)
plt.title('Distribution of Reconstruction Errors')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.savefig('reconstruction_error_distribution.png')
plt.close()

threshold = np.percentile(reconstruction_errors.cpu().numpy(), 95)  # Set threshold at 95th percentile
print(f"Threshold for anomalies: {threshold}")

anomalous = reconstruction_errors > threshold
normal = reconstruction_errors <= threshold

def visualize_anomalous(normal_images, anomalous_images, nrow=8, title=''):
    normal_grid = torchvision.utils.make_grid(normal_images, nrow=nrow, normalize=True)
    anomalous_grid = torchvision.utils.make_grid(anomalous_images, nrow=nrow, normalize=True)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(normal_grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.title('Normal Images')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(anomalous_grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.title('Anomalous Images')
    plt.axis('off')
    
    plt.savefig(f'anomalous_vs_normal.png')
    plt.close()

normal_images = noisy_images[normal].view(-1, 1, 28, 28)
anomalous_images = noisy_images[anomalous].view(-1, 1, 28, 28)

visualize_anomalous(normal_images, anomalous_images)
