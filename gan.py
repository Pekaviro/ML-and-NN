import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F

RANDOM_SEED = 42
IMG_CHANNELS = 1

# Воспроизводимость
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # Для 1 канала FashionMNIST
])

dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)  # ← num_workers=0 для Windows

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=IMG_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_channels=IMG_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, 4, 2, 1, bias=False)), 
                nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(512, 1, 4, 1, 0, bias=False)),
        )
    def forward(self, img):
        return self.net(img).view(-1)

def denorm_gan(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def save_grid(images, epoch, path='grids'):
    import os
    os.makedirs(path, exist_ok=True)
    images = denorm_gan(images).cpu()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.savefig(f'{path}/epoch_{epoch}.png')
    plt.close()

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def prepare_for_fid(images):
    """Подготовка изображений для FID: 1 канал → 3 канала, ресайз до 299x299"""
    images = denorm_gan(images)  # [0, 1]
    
    # Конвертируем 1 канал в 3 канала (повторяем канал 3 раза)
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)  # [batch, 1, H, W] → [batch, 3, H, W]
    
    # InceptionV3 требует размер 299x299
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    
    return (images * 255).byte()  # [0, 255] uint8

# ОБУЧЕНИЕ
if __name__ == '__main__':
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()

    opt_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, latent_dim, 1, 1, device=device)
    num_epochs = 50
    history = {'loss_D': [], 'loss_G': [], 'D_x': [], 'D_G_z': []}

    for epoch in range(num_epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        epoch_D_x = 0.0
        epoch_D_G_z = 0.0

        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            real_labels = torch.full((batch_size,), 0.9, device=device)
            fake_labels = torch.zeros(batch_size, device=device)
            
            # Шаг 1: обучение дискриминатора
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images.detach())
            loss_real = criterion(d_real, real_labels)
            loss_fake = criterion(d_fake, fake_labels)
            loss_D = loss_real + loss_fake
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            
            # Шаг 2: обучение генератора
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            d_fake_for_G = discriminator(fake_images)
            loss_G = criterion(d_fake_for_G, torch.ones(batch_size, device=device))
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            epoch_D_x += torch.sigmoid(d_real).mean().item()
            epoch_D_G_z += torch.sigmoid(d_fake_for_G).mean().item()

        n = len(dataloader)
        history['loss_D'].append(epoch_loss_D / n)
        history['loss_G'].append(epoch_loss_G / n)
        history['D_x'].append(epoch_D_x / n)
        history['D_G_z'].append(epoch_D_G_z / n)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"loss_D={history['loss_D'][-1]:.4f} "
              f"loss_G={history['loss_G'][-1]:.4f} "
              f"D(x)={history['D_x'][-1]:.3f} "
              f"D(G(z))={history['D_G_z'][-1]:.3f}")

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise)
                save_grid(fake, epoch+1)

    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['loss_D'], label='Discriminator Loss', color='red')
    ax1.plot(history['loss_G'], label='Generator Loss', color='blue')
    ax1.axhline(y=0.693, color='green', linestyle='--', label='ln(2) ≈ 0.693')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Losses over epochs')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['D_x'], label='D(x) - real', color='green')
    ax2.plot(history['D_G_z'], label='D(G(z)) - fake', color='orange')
    ax2.axhline(y=0.5, color='black', linestyle='--', label='0.5')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Probability')
    ax2.set_title('Discriminator output over epochs')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()

    # Финальная генерация 64 изображений
    final_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        final_images = generator(final_noise)
        final_images = denorm_gan(final_images).cpu()

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        img = final_images[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Final generated FashionMNIST images (64 samples)')
    plt.savefig('final_grid.png', dpi=150)
    plt.show()

    # FID метрика
    NUM_FID_SAMPLES = 10_000
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Реальные изображения
    samples_seen = 0
    for real_images, _ in dataloader:
        real_images = real_images.to(device)
        fid.update(prepare_for_fid(real_images), real=True)
        samples_seen += real_images.size(0)
        if samples_seen >= NUM_FID_SAMPLES:
            break

    # Сгенерированные изображения
    generator.eval()
    samples_seen = 0
    with torch.no_grad():
        while samples_seen < NUM_FID_SAMPLES:
            z = torch.randn(128, latent_dim, 1, 1, device=device)
            fake_images = generator(z)
            fid.update(prepare_for_fid(fake_images), real=False)
            samples_seen += fake_images.size(0)

    print(f"FID: {fid.compute():.2f}")

    # Интерполяция между векторами шума
    z1 = torch.randn(1, latent_dim, 1, 1, device=device)
    z2 = torch.randn(1, latent_dim, 1, 1, device=device)
    alphas = torch.linspace(0, 1, steps=10)

    with torch.no_grad():
        images = [generator((1 - a) * z1 + a * z2) for a in alphas]

    # ВИЗУАЛИЗАЦИЯ ИНТЕРПОЛЯЦИИ
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, (ax, img) in enumerate(zip(axes.flat, images)):
        img_disp = denorm_gan(img).cpu().squeeze().numpy()
        ax.imshow(img_disp, cmap='gray')
        ax.set_title(f'α={alphas[i]:.1f}', fontsize=12)
        ax.axis('off')
    plt.suptitle('Latent Space Interpolation: z1 → z2', fontsize=14)
    plt.tight_layout()
    plt.savefig('interpolation.png', dpi=150)
    plt.show()