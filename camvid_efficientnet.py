# camvid_efficientnet.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import gc
from torch.amp import autocast, GradScaler

SIZE = 224
NUM_CLASSES = 32

NUM_EPOCHS = 50
LEARNING_RATE_ENCODER = 1e-5  # маленький lr для предобученного encoder
LEARNING_RATE_DECODER = 1e-4  # нормальный lr для нового decoder
WEIGHT_DECAY = 1e-4

# Путь к папке с изображениями
train_path = "C:/Users/user/.cache/kagglehub/datasets/carlolepelaars/camvid/versions/2/CamVid/train"

# Вычисление mean и std
transform = transforms.Compose([transforms.ToTensor()])

mean = 0.0
std = 0.0
total = 0

for img_name in os.listdir(train_path):
    if img_name.endswith(('.png', '.jpg')):
        img_path = os.path.join(train_path, img_name)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        mean += img_tensor.mean(dim=[1,2])
        std += img_tensor.std(dim=[1,2])
        total += 1

mean /= total
std /= total

print(f"Mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
print(f"Std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")

# Трансформации с нормализацией ImageNet (для EfficientNet)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform_train = A.Compose([
    A.Resize(SIZE, SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ToTensorV2(),
])

transform_val = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ToTensorV2(),
])

# Класс Dataset для CamVid
class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_dict_path, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(images_dir) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) 
                              if f.endswith('.png')])
        
        assert len(self.images) == len(self.masks), \
            f"Количество изображений ({len(self.images)}) не совпадает с масками ({len(self.masks)})"
        
        self.color_to_idx = self.load_class_mapping(class_dict_path)
        
        print(f"Загружено {len(self.images)} пар из {os.path.basename(images_dir)}")
    
    def load_class_mapping(self, class_dict_path):
        df = pd.read_csv(class_dict_path)
        color_to_idx = {}
        
        for idx, row in df.iterrows():
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_idx[(r, g, b)] = idx
        
        print(f"Загружено {len(color_to_idx)} классов")
        return color_to_idx
    
    def rgb_to_class_mask(self, rgb_mask):
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        for color, idx in self.color_to_idx.items():
            mask = (rgb_mask[:, :, 0] == color[0]) & \
                   (rgb_mask[:, :, 1] == color[1]) & \
                   (rgb_mask[:, :, 2] == color[2])
            class_mask[mask] = idx
        
        return class_mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        mask = self.rgb_to_class_mask(mask)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        
        return image, mask

# U-Net с предобученным EfficientNet-B0 encoder
class CamVidEfficientNet(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        
        # Загружаем предобученный EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Encoder уровни
        # Уровень 1: после conv stem (3 -> 32)
        self.enc1 = nn.Sequential(
            self.backbone.features[0]
        )
        
        # Уровень 2: после 2-го блока (32 -> 24)
        self.enc2 = nn.Sequential(
            self.backbone.features[1],
            self.backbone.features[2]
        )
        
        # Уровень 3: после 3-го блока (24 -> 40)
        self.enc3 = nn.Sequential(
            self.backbone.features[3]
        )
        
        # Уровень 4: после 4-го и 5-го блоков (40 -> 112)
        self.enc4 = nn.Sequential(
            self.backbone.features[4],
            self.backbone.features[5]
        )
        
        # Уровень 5: после 6-го и 7-го блоков (112 -> 320)
        self.enc5 = nn.Sequential(
            self.backbone.features[6],
            self.backbone.features[7]
        )
        
        # Bottleneck
        self.bottleneck = self._block(320, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 112, kernel_size=2, stride=2)
        self.dec4 = self._block(112 + 112, 112)
        
        self.up3 = nn.ConvTranspose2d(112, 40, kernel_size=2, stride=2)
        self.dec3 = self._block(40 + 40, 40)
        
        self.up2 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)
        self.dec2 = self._block(24 + 24, 24)
        
        self.up1 = nn.ConvTranspose2d(24, 32, kernel_size=2, stride=2)
        self.dec1 = self._block(32 + 32, 32)
        
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        b = self.bottleneck(e5)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_up(d1)
        out = self.out_conv(out)
        
        return out

# Проверка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = CamVidEfficientNet(num_classes=NUM_CLASSES)
model = model.to(device)

# Тестовый forward
x = torch.randn(1, 3, SIZE, SIZE).to(device)
y = model(x)
print(f"Input shape: {x.shape} → Output shape: {y.shape}")

# Создание датасетов
data_root = "C:/Users/user/.cache/kagglehub/datasets/carlolepelaars/camvid/versions/2/CamVid"
class_dict_path = os.path.join(data_root, "class_dict.csv")

if not os.path.exists(class_dict_path):
    print(f"class_dict.csv не найден в {data_root}")
    exit()

train_dataset = CamVidDataset(
    images_dir=os.path.join(data_root, "train"),
    masks_dir=os.path.join(data_root, "train_labels"),
    class_dict_path=class_dict_path,
    transform=transform_train
)

val_dataset = CamVidDataset(
    images_dir=os.path.join(data_root, "val"),
    masks_dir=os.path.join(data_root, "val_labels"),
    class_dict_path=class_dict_path,
    transform=transform_val
)

test_dataset = CamVidDataset(
    images_dir=os.path.join(data_root, "test"),
    masks_dir=os.path.join(data_root, "test_labels"),
    class_dict_path=class_dict_path,
    transform=transform_val
)

BATCH_SIZE = 8

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nBatch size: {BATCH_SIZE}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Разделяем параметры на encoder и decoder
encoder_params = []
decoder_params = []

for name, param in model.named_parameters():
    if 'backbone' in name:  # параметры предобученного encoder
        encoder_params.append(param)
    else:  # параметры decoder
        decoder_params.append(param)

# Оптимизатор с разными learning rates
optimizer = optim.Adam([
    {'params': encoder_params, 'lr': LEARNING_RATE_ENCODER},
    {'params': decoder_params, 'lr': LEARNING_RATE_DECODER},
], weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

# Функции для вычисления метрик
def compute_iou(pred_mask, true_mask, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0).to(device)
        iou_per_class.append(iou.item())
    mean_iou = np.mean([iou for iou in iou_per_class if iou > 0])
    return iou_per_class, mean_iou

def compute_dice_score(pred_mask, true_mask, num_classes):
    dice_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        intersection = (pred_cls & true_cls).sum().float()
        pred_sum = pred_cls.sum().float()
        true_sum = true_cls.sum().float()
        if pred_sum + true_sum > 0:
            dice = 2 * intersection / (pred_sum + true_sum)
        else:
            dice = torch.tensor(0.0).to(device)
        dice_per_class.append(dice.item())
    mean_dice = np.mean([dice for dice in dice_per_class if dice > 0])
    return dice_per_class, mean_dice

def validate(model, val_loader, device, num_classes):
    model.eval()
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            pred_masks = logits.argmax(dim=1)
            
            batch_iou = 0
            for i in range(images.size(0)):
                _, iou_val = compute_iou(pred_masks[i], masks[i], num_classes)
                batch_iou += iou_val
            
            total_iou += batch_iou / images.size(0)
            num_batches += 1
    
    return total_iou / num_batches

def test_model(model, test_loader, device, num_classes):
    model.eval()
    total_iou = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            pred_masks = logits.argmax(dim=1)
            
            batch_iou = 0
            batch_dice = 0
            for i in range(images.size(0)):
                _, iou_val = compute_iou(pred_masks[i], masks[i], num_classes)
                _, dice_val = compute_dice_score(pred_masks[i], masks[i], num_classes)
                batch_iou += iou_val
                batch_dice += dice_val
            
            total_iou += batch_iou / images.size(0)
            total_dice += batch_dice / images.size(0)
            num_batches += 1
    
    return total_iou / num_batches, total_dice / num_batches

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Обучение модели одну эпоху
    Returns:
        avg_loss: средний loss за эпоху
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    print(f"  Обучение: ", end="")
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(images)
            loss = criterion(logits, masks)
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    avg_loss = total_loss / num_batches
    return avg_loss

if __name__ == "__main__":
    # Основной цикл обучения
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ U-NET С EFFICIENTNET-B0 ENCODER")
    print("="*60)
    print(f"\nLearning rates:")
    print(f"  Encoder (pretrained): {LEARNING_RATE_ENCODER}")
    print(f"  Decoder (new): {LEARNING_RATE_DECODER}")

    train_losses = []
    val_ious = []
    best_val_iou = 0
    patience = 10
    patience_counter = 0
    stop_epoch = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Эпоха {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        print(f"  Валидация: ", end="", flush=True)
        val_iou = validate(model, val_loader, device, NUM_CLASSES)
        print("Готово")
        val_ious.append(val_iou)
        
        print(f"\nРезультаты эпохи {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_iou:.4f}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'train_loss': train_loss
            }, 'best_model_efficientnet_unet.pth')
            print(f"  ✓ Новая лучшая модель сохранена! (mIoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            stop_epoch = epoch
            print(f"\nРанняя остановка на эпохе {epoch}")
            break

    if stop_epoch == 0:
        stop_epoch = NUM_EPOCHS

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2)
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Потери')
    axes[0].set_title('Функция потерь')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_ious, label='Val mIoU', marker='s', markersize=3, linewidth=2, color='red')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('Mean IoU на валидации')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics_efficientnet.png', dpi=150)
    plt.show()

    # Тестирование
    checkpoint = torch.load('best_model_efficientnet_unet.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_iou, test_dice = test_model(model, test_loader, device, NUM_CLASSES)

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*60)
    print(f"Mean IoU: {test_iou:.4f}")
    print(f"Mean Dice Score: {test_dice:.4f}")

    # Сохранение результатов
    with open('camvid_efficientnet_results.txt', 'w') as f:
        f.write("РЕЗУЛЬТАТЫ U-NET С EFFICIENTNET-B0 ENCODER\n")
        f.write("="*60 + "\n\n")
        f.write(f"Эпоха остановки: {stop_epoch}\n")
        f.write(f"Learning rate encoder: {LEARNING_RATE_ENCODER}\n")
        f.write(f"Learning rate decoder: {LEARNING_RATE_DECODER}\n\n")
        f.write(f"Лучший Val mIoU: {best_val_iou:.4f} (эпоха {checkpoint['epoch']})\n")
        f.write(f"Test mIoU: {test_iou:.4f}\n")
        f.write(f"Test Dice: {test_dice:.4f}\n")