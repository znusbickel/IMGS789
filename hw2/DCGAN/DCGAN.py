import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100
num_epochs = 25
batch_size = 128
learning_rate = 0.0002

transform = transforms.Compose([
    transforms.Resize(64),  # Resize images to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),  # 1x1 -> 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh()  # Output range is [-1, 1]
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 4x4 -> 1x1
        )

    def forward(self, img):
        return self.model(img).view(-1)  # Flatten to [batch_size]

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()

for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)

        labels_real = torch.ones(real_images.size(0), device=device)  # Real labels
        labels_fake = torch.zeros(real_images.size(0), device=device)  # Fake labels

        optimizer_D.zero_grad()
        
        outputs_real = discriminator(real_images)
        d_loss_real = criterion(outputs_real, labels_real)

        noise = torch.randn(real_images.size(0), latent_dim, 1, 1, device=device)
        fake_images = generator(noise)  # Generate fake images
        outputs_fake = discriminator(fake_images.detach())  # Detach to not update the generator
        d_loss_fake = criterion(outputs_fake, labels_fake)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)  # Output for fake images
        g_loss = criterion(outputs, labels_real)  # We want the generator to fool the discriminator
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

def show_generated_images(generator, num_images=64):
    with torch.no_grad():
        generator.eval()
        z = torch.randn(num_images, latent_dim, 1, 1, device=device)
        generated_images = generator(z).cpu().detach()
        generated_images = (generated_images + 1) / 2  # Rescale to [0, 1]

        grid = torchvision.utils.make_grid(generated_images, nrow=8, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title('Generated Images')
        plt.savefig('generator_im.png')

show_generated_images(generator)

def show_real_images(train_loader):
    real_images, _ = next(iter(train_loader))
    grid = torchvision.utils.make_grid(real_images, nrow=8, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
    plt.axis('off')
    plt.title('Real CIFAR-10 Images')
    plt.savefig('CIFAR_im.png')

show_real_images(train_loader)
