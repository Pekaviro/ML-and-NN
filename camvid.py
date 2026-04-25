# C:\Users\user\.cache\kagglehub\datasets\carlolepelaars\camvid\versions\2
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
import gc
from torch.amp import autocast, GradScaler


SIZE = 224
NUM_CLASSES = 32

NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Путь к папке с изображениями
train_path = "C:/Users/user/.cache/kagglehub/datasets/carlolepelaars/camvid/versions/2/CamVid/train"




# Трансформация: только в тензор
transform = transforms.Compose([transforms.ToTensor()])

# Загружаем все изображения вручную
mean = 0.0
std = 0.0
total = 0

for img_name in os.listdir(train_path):
    if img_name.endswith(('.png', '.jpg')):
        # Загружаем изображение
        img_path = os.path.join(train_path, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Применяем трансформацию
        img_tensor = transform(img)  # форма: [3, H, W]

        # Накопливаем статистику
        mean += img_tensor.mean(dim=[1,2])
        std += img_tensor.std(dim=[1,2])
        total += 1

# Усредняем
mean /= total
std /= total

print(f"Mean: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
print(f"Std: [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")


transform_train = A.Compose([
    A.Resize(SIZE, SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean, std),
    ToTensorV2(),
    ])
transform_val = A.Compose([
    A.Resize(SIZE, SIZE),
    A.Normalize(mean,std),
    ToTensorV2(),
    ])


# Класс Dataset для CamVid
class CamVidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_dict_path, transform=None):
        """
        Args:
            images_dir: путь к папке с изображениями (train, val, или test)
            masks_dir: путь к папке с масками (train_labels, val_labels, или test_labels)
            class_dict_path: путь к файлу class_dict.csv
            transform: аугментации от albumentations
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # Получаем списки файлов
        self.images = sorted([f for f in os.listdir(images_dir) 
                              if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.masks = sorted([f for f in os.listdir(masks_dir) 
                              if f.endswith('.png')])
        
        # Проверяем соответствие
        assert len(self.images) == len(self.masks), \
            f"Количество изображений ({len(self.images)}) не совпадает с масками ({len(self.masks)})"
        
        # Загружаем соответствие цветов классам
        self.color_to_idx = self.load_class_mapping(class_dict_path)
        
        print(f"Загружено {len(self.images)} пар из {os.path.basename(images_dir)}")
    
    def load_class_mapping(self, class_dict_path):
        """Загружает соответствие RGB -> индекс класса из class_dict.csv"""
        df = pd.read_csv(class_dict_path)
        color_to_idx = {}
        
        for idx, row in df.iterrows():
            r, g, b = int(row['r']), int(row['g']), int(row['b'])
            color_to_idx[(r, g, b)] = idx
        
        print(f"Загружено {len(color_to_idx)} классов")
        return color_to_idx
    
    def rgb_to_class_mask(self, rgb_mask):
        """Преобразует цветную маску (H, W, 3) в маску с индексами классов (H, W)"""
        h, w = rgb_mask.shape[:2]
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        # Для каждого цвета назначаем индекс класса
        for color, idx in self.color_to_idx.items():
            mask = (rgb_mask[:, :, 0] == color[0]) & \
                   (rgb_mask[:, :, 1] == color[1]) & \
                   (rgb_mask[:, :, 2] == color[2])
            class_mask[mask] = idx
        
        return class_mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загружаем маску (цветная)
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        
        # Преобразуем цветную маску в индексы классов
        mask = self.rgb_to_class_mask(mask)
        
        # Применяем аугментации
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        
        return image, mask
    


# Создание всех датасетов и загрузчиков 
data_root = "C:/Users/user/.cache/kagglehub/datasets/carlolepelaars/camvid/versions/2/CamVid"
class_dict_path = os.path.join(data_root, "class_dict.csv")

# Проверяем существование class_dict.csv
if not os.path.exists(class_dict_path):
    print(f"class_dict.csv не найден в {data_root}")
    exit()

# Создаем датасеты для каждого сплита
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

# Создаем DataLoader'ы
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# image, mask = train_dataset[0]
# print(f"image shape: {image.shape}") # ожидается: [3, 128, 128]
# print(f"mask shape: {mask.shape}") # ожидается: [128, 128]
# print(f"mask dtype: {mask.dtype}") # ожидается: torch.int64
# print(f"mask classes: {mask.unique()}") # значения от 0 до num_classes-1




# Функция денормализации
def denormalize(image_tensor):
    mean_tensor = mean.clone().detach().view(3, 1, 1)
    std_tensor = std.clone().detach().view(3, 1, 1)
    img = image_tensor * std_tensor + mean_tensor
    img = torch.clamp(img, 0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


random_indices = random.sample(range(len(train_dataset)), 6)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))

for i in range(3):
    for j in range(2):
        idx = random_indices[i * 2 + j]
        image, mask = train_dataset[idx]
        
        # Изображение
        img_display = denormalize(image)
        axes[i, j * 2].imshow(img_display)
        axes[i, j * 2].axis('off')
        
        # Маска с простой цветовой картой
        axes[i, j * 2 + 1].imshow(mask.numpy(), cmap='nipy_spectral', interpolation='none')
        axes[i, j * 2 + 1].axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()




class CamVidNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ===== Encoder =====
        self.enc1 = self._block(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # ===== Bottleneck =====
        self.bottleneck = self._block(256, 512)

        # ===== Decoder =====
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(256 + 256, 256) # skip-connection
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(64 + 64, 64)

        # ===== Output =====
        self.out_conv = nn.Conv2d(64, NUM_CLASSES, kernel_size=1)

    def _block(self, in_ch, out_ch):
        """свёртка: Conv → BN → ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # ===== Encoder =====
        e1 = self.enc1(x) # [B, 64, 256, 256]
        e2 = self.enc2(self.pool1(e1)) # [B, 128, 128, 128]
        e3 = self.enc3(self.pool2(e2)) # [B, 256, 64, 64]

        # ===== Bottleneck =====
        b = self.bottleneck(self.pool3(e3)) # [B, 512, 32, 32]

        # ===== Decoder =====
        d3 = self.up3(b) # [B, 256, 64, 64]
        d3 = torch.cat([d3, e3], dim=1) # skip-connection
        d3 = self.dec3(d3)
        d2 = self.up2(d3) # [B, 128, 128, 128]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2) # [B, 64, 256, 256]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)
    

model = CamVidNet()
x = torch.randn(1, 3, 256, 256)
y = model(x)
print(x.shape, "→", y.shape)





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CamVidNet()
model = model.to(device)

scaler = GradScaler()

# Функция для вычисления IoU
def compute_iou(pred_mask, true_mask, num_classes):
    """
    Вычисляет IoU для каждого класса
    Args:
        pred_mask: [H, W] предсказанные индексы классов
        true_mask: [H, W] истинные индексы классов
        num_classes: количество классов
    Returns:
        iou_per_class: список IoU для каждого класса
        mean_iou: средний IoU
    """
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

# Функция для вычисления Dice Score (только для теста)
def compute_dice_score(pred_mask, true_mask, num_classes):
    """
    Вычисляет Dice Score для каждого класса
    Args:
        pred_mask: [H, W] предсказанные индексы классов
        true_mask: [H, W] истинные индексы классов
        num_classes: количество классов
    Returns:
        dice_per_class: список Dice для каждого класса
        mean_dice: средний Dice
    """
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

# Функция для валидации (только IoU)
def validate(model, val_loader, device, num_classes):
    """
    Валидация модели на валидационной выборке
    Returns:
        mean_iou: средний IoU на валидации
    """
    model.eval()
    total_iou = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Получаем предсказания
            logits = model(images)
            pred_masks = logits.argmax(dim=1)  # [B, H, W]
            
            # Вычисляем метрики для каждого изображения в батче
            batch_iou = 0
            
            for i in range(images.size(0)):
                _, iou_val = compute_iou(pred_masks[i], masks[i], num_classes)
                batch_iou += iou_val
            
            total_iou += batch_iou / images.size(0)
            num_batches += 1
    
    mean_iou = total_iou / num_batches
    
    return mean_iou

# Функция для тестирования (IoU и Dice)
def test_model(model, test_loader, device, num_classes):
    """
    Тестирование модели на тестовой выборке
    Returns:
        mean_iou: средний IoU на тесте
        mean_dice: средний Dice на тесте
    """
    model.eval()
    total_iou = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Получаем предсказания
            logits = model(images)
            pred_masks = logits.argmax(dim=1)  # [B, H, W]
            
            # Вычисляем метрики для каждого изображения в батче
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
    
    mean_iou = total_iou / num_batches
    mean_dice = total_dice / num_batches
    
    return mean_iou, mean_dice

# Функция для обучения одной эпохи
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs):
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
    # Основной скрипт обучения
    print("="*60)
    print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ СЕГМЕНТАЦИИ CAMVID")
    print("="*60)


    # Оптимизатор Adam
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Функция потерь
    criterion = nn.CrossEntropyLoss()

    # Списки для сохранения метрик
    train_losses = []
    val_ious = []
    epochs_list = []

    best_val_iou = 0
    patience = 10
    patience_counter = 0
    stop_epoch = 0

    print(f"\nПараметры обучения:")
    print(f"  Эпох: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Device: {device}")
    print(f"  Количество классов: {NUM_CLASSES}")
    print(f"  Размер батча: {train_loader.batch_size}")
    print(f"\nНачинаем обучение...\n")

    # Цикл обучения
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Эпоха {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Обучение
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, NUM_EPOCHS)
        train_losses.append(train_loss)
        
        # Валидация (только IoU)
        print(f"  Валидация: ", end="", flush=True)
        val_iou = validate(model, val_loader, device, NUM_CLASSES)
        print("Готово")
        val_ious.append(val_iou)
        epochs_list.append(epoch)
        
        # Вывод результатов
        print(f"\nРезультаты эпохи {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mIoU: {val_iou:.4f}")
        
        # Сохранение лучшей модели по mIoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'train_loss': train_loss
            }, 'best_model_camvid.pth')
            print(f"  ✓ Новая лучшая модель сохранена! (mIoU: {val_iou:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Ранняя остановка
        if patience_counter >= patience:
            stop_epoch = epoch
            print(f"\nРанняя остановка на эпохе {epoch}")
            break
        
        print("-"*60)

    # Если обучение завершилось без ранней остановки
    if stop_epoch == 0:
        stop_epoch = NUM_EPOCHS

    # Визуализация графиков
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ ГРАФИКОВ ОБУЧЕНИЯ")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График функции потерь
    axes[0].plot(epochs_list[:len(train_losses)], train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2, color='blue')
    axes[0].set_xlabel('Эпоха', fontsize=12)
    axes[0].set_ylabel('Потери', fontsize=12)
    axes[0].set_title('Изменение функции потерь', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # График mIoU на валидации
    axes[1].plot(epochs_list[:len(val_ious)], val_ious, label='Val mIoU', marker='s', markersize=3, linewidth=2, color='red')
    axes[1].set_xlabel('Эпоха', fontsize=12)
    axes[1].set_ylabel('mIoU', fontsize=12)
    axes[1].set_title('Mean IoU на валидации', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Визуализация предсказаний на валидационной выборке
    print("\n" + "="*60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ НА ВАЛИДАЦИОННОЙ ВЫБОРКЕ")
    print("="*60)

    # Загружаем лучшую модель
    checkpoint = torch.load('best_model_camvid.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Визуализация
    model.eval()
    num_samples = 6
    random_indices = random.sample(range(len(val_dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))

    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            image, true_mask = val_dataset[idx]
            image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Предсказание
            logits = model(image)
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu()  # [H, W]
            
            # Денормализация изображения для отображения
            img_display = denormalize(image.squeeze(0).cpu())
            
            # Исходное изображение
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Исходное изображение', fontsize=10)
            axes[i, 0].axis('off')
            
            # Истинная маска
            axes[i, 1].imshow(true_mask.numpy(), cmap='nipy_spectral', interpolation='none')
            axes[i, 1].set_title('Истинная маска', fontsize=10)
            axes[i, 1].axis('off')
            
            # Предсказанная маска
            axes[i, 2].imshow(pred_mask.numpy(), cmap='nipy_spectral', interpolation='none')
            axes[i, 2].set_title('Предсказанная маска', fontsize=10)
            axes[i, 2].axis('off')

    plt.suptitle('Сравнение истинных и предсказанных масок', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Оценка на тестовой выборке (IoU и Dice)
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*60)

    print("  Тестирование: ", end="", flush=True)
    test_iou, test_dice = test_model(model, test_loader, device, NUM_CLASSES)
    print("Готово")

    print(f"\nРезультаты на тестовой выборке:")
    print(f"  Mean IoU: {test_iou:.4f}")
    print(f"  Mean Dice Score: {test_dice:.4f}")
    print("="*60)

    # Сохранение результатов в файл
    with open('camvid_results.txt', 'w', encoding='utf-8') as f:
        f.write("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛИ CAMVID\n")
        f.write("="*60 + "\n\n")
        f.write(f"Параметры обучения:\n")
        f.write(f"  Максимальное количество эпох: {NUM_EPOCHS}\n")
        f.write(f"  Эпоха остановки: {stop_epoch}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n")
        f.write(f"  Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"  Оптимизатор: Adam\n")
        f.write(f"  Функция потерь: CrossEntropyLoss\n")
        f.write(f"  Early stopping patience: {patience}\n\n")
        
        f.write(f"Лучший результат на валидации:\n")
        f.write(f"  Эпоха: {checkpoint['epoch']}\n")
        f.write(f"  Mean IoU: {checkpoint['val_iou']:.4f}\n\n")
        
        f.write(f"Результаты на тестовой выборке:\n")
        f.write(f"  Mean IoU: {test_iou:.4f}\n")
        f.write(f"  Mean Dice Score: {test_dice:.4f}\n")

    print("\n✅ Результаты сохранены в файл 'camvid_results.txt'")
    print("✅ Графики сохранены в 'training_metrics.png'")
    print("✅ Визуализация предсказаний сохранена в 'predictions_visualization.png'")
    print("✅ Лучшая модель сохранена в 'best_model_camvid.pth'")

    # Вывод финальной статистики
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Обучение завершено на эпохе: {stop_epoch}")
    print(f"Лучший mIoU на валидации: {checkpoint['val_iou']:.4f} (эпоха {checkpoint['epoch']})")
    print(f"Тестовый mIoU: {test_iou:.4f}")
    print(f"Тестовый Dice Score: {test_dice:.4f}")
    print("="*60)