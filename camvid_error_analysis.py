import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pandas as pd
from torch.utils.data import DataLoader
import cv2

# Импортируем нужные компоненты
from camvid import (
    CamVidDataset, test_dataset, device, NUM_CLASSES, denormalize, transform_val
)
from camvid_efficientnet import CamVidEfficientNet, transform_val as transform_val_eff

# Пути
data_root = "C:/Users/user/.cache/kagglehub/datasets/carlolepelaars/camvid/versions/2/CamVid"
class_dict_path = os.path.join(data_root, "class_dict.csv")

# Загружаем названия классов
df_classes = pd.read_csv(class_dict_path)
class_names = df_classes['name'].tolist()

# Создаем тестовый датасет
test_dataset = CamVidDataset(
    images_dir=os.path.join(data_root, "test"),
    masks_dir=os.path.join(data_root, "test_labels"),
    class_dict_path=class_dict_path,
    transform=transform_val_eff
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# Загружаем модель
print("Загрузка модели...")
model = CamVidEfficientNet(num_classes=NUM_CLASSES)
model = model.to(device)

if os.path.exists('best_model_efficientnet_unet.pth'):
    checkpoint = torch.load('best_model_efficientnet_unet.pth', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель загружена (эпоха {checkpoint['epoch']}, val mIoU: {checkpoint['val_iou']:.4f})")
else:
    print("Модель не найдена!")
    exit()

# Функция для вычисления IoU для одного изображения
def compute_iou_single(pred_mask, true_mask, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        
        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0)
        
        iou_per_class.append(iou.item())
    
    mean_iou = np.mean([iou for iou in iou_per_class if iou > 0])
    return mean_iou, iou_per_class

# Функция для анализа ошибок по классам
def analyze_class_errors(model, test_loader, device, num_classes, class_names):
    model.eval()
    class_iou_sum = [0.0] * num_classes
    class_count = [0] * num_classes
    class_error_pixels = [0] * num_classes
    class_total_pixels = [0] * num_classes
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            pred_masks = logits.argmax(dim=1)
            
            for i in range(images.size(0)):
                for cls in range(num_classes):
                    true_cls = (masks[i] == cls)
                    pred_cls = (pred_masks[i] == cls)
                    
                    # IoU
                    intersection = (pred_cls & true_cls).sum().float()
                    union = (pred_cls | true_cls).sum().float()
                    
                    if union > 0:
                        iou = intersection / union
                        class_iou_sum[cls] += iou.item()
                        class_count[cls] += 1
                    
                    # Ошибки
                    class_total_pixels[cls] += true_cls.sum().item()
                    error_pixels = (pred_cls != true_cls).sum().item()
                    class_error_pixels[cls] += error_pixels
    
    class_ious = []
    class_error_rates = []
    
    for cls in range(num_classes):
        if class_count[cls] > 0:
            class_ious.append(class_iou_sum[cls] / class_count[cls])
        else:
            class_ious.append(0.0)
        
        if class_total_pixels[cls] > 0:
            error_rate = class_error_pixels[cls] / class_total_pixels[cls]
            class_error_rates.append(error_rate)
        else:
            class_error_rates.append(0.0)
    
    return class_ious, class_error_rates

# Функция для визуализации ошибок
def visualize_errors(model, test_loader, device, num_samples=6):
    model.eval()
    
    # Собираем все результаты
    results = []
    
    with torch.no_grad():
        for idx, (image, true_mask) in enumerate(test_loader):
            image = image.to(device)
            true_mask = true_mask.to(device)
            
            logits = model(image)
            pred_mask = logits.argmax(dim=1)
            
            # Вычисляем IoU для этого изображения
            iou, iou_per_class = compute_iou_single(pred_mask[0], true_mask[0], NUM_CLASSES)
            
            results.append({
                'idx': idx,
                'image': image.cpu(),
                'true_mask': true_mask.cpu(),
                'pred_mask': pred_mask.cpu(),
                'iou': iou,
                'iou_per_class': iou_per_class
            })
    
    # Сортируем по IoU (худшие первые)
    results.sort(key=lambda x: x['iou'])
    
    # Берем num_samples худших
    worst_samples = results[:num_samples]
    
    # Функция денормализации для ImageNet
    def denormalize_imagenet(image_tensor):
        mean_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_tensor = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = image_tensor * std_tensor + mean_tensor
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        return img
    
    # Визуализация
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    for i, sample in enumerate(worst_samples):
        # Исходное изображение
        img_display = denormalize_imagenet(sample['image'].squeeze(0))
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f'Исходное изображение\nIoU: {sample["iou"]:.3f}', fontsize=10)
        axes[i, 0].axis('off')
        
        # Истинная маска
        axes[i, 1].imshow(sample['true_mask'].squeeze(0).numpy(), cmap='nipy_spectral', interpolation='none')
        axes[i, 1].set_title('Истинная маска', fontsize=10)
        axes[i, 1].axis('off')
        
        # Предсказанная маска
        axes[i, 2].imshow(sample['pred_mask'].squeeze(0).numpy(), cmap='nipy_spectral', interpolation='none')
        axes[i, 2].set_title('Предсказанная маска', fontsize=10)
        axes[i, 2].axis('off')
        
        # Карта ошибок
        error_map = (sample['pred_mask'].squeeze(0) != sample['true_mask'].squeeze(0)).float()
        axes[i, 3].imshow(error_map.numpy(), cmap='Reds', interpolation='none')
        axes[i, 3].set_title(f'Карта ошибок\n{error_map.sum().item():.0f} ошибочных пикселей', fontsize=10)
        axes[i, 3].axis('off')
    
    plt.suptitle('Примеры изображений с наибольшими ошибками', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('error_analysis_worst_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return worst_samples

# Функция для анализа типов ошибок
def analyze_error_types(model, test_loader, device, class_names):
    model.eval()
    
    # Статистика ошибок
    boundary_errors = 0
    small_object_errors = 0
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    total_boundary_pixels = 0
    total_small_objects = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            logits = model(images)
            pred_masks = logits.argmax(dim=1)
            
            for i in range(images.size(0)):
                true_mask = masks[i].cpu().numpy()
                pred_mask = pred_masks[i].cpu().numpy()
                
                # 1. Ошибки на границах
                # Находим границы объектов в истинной маске
                from scipy import ndimage
                boundaries = np.zeros_like(true_mask)
                for cls in range(NUM_CLASSES):
                    class_mask = (true_mask == cls).astype(np.uint8)
                    if class_mask.sum() > 0:
                        # Эрозия для нахождения границ
                        eroded = ndimage.binary_erosion(class_mask, structure=np.ones((3,3)))
                        boundary = class_mask & ~eroded
                        boundaries[boundary] = 1
                        total_boundary_pixels += boundary.sum()
                        
                        # Считаем ошибки на границах
                        boundary_errors += ((pred_mask != true_mask) & boundary).sum()
                
                # 2. Ошибки на мелких объектах
                for cls in range(NUM_CLASSES):
                    class_mask = (true_mask == cls)
                    class_size = class_mask.sum()
                    
                    # Определяем мелкие объекты (< 1000 пикселей при размере 224x224)
                    if 0 < class_size < 1000:
                        total_small_objects += 1
                        # Проверяем, правильно ли предсказан объект
                        pred_class_mask = (pred_mask == cls)
                        intersection = (class_mask & pred_class_mask).sum()
                        if intersection < class_size * 0.5:  # <50% покрытия
                            small_object_errors += 1
                
                # 3. Матрица ошибок (путаница классов)
                for h in range(true_mask.shape[0]):
                    for w in range(true_mask.shape[1]):
                        true_class = true_mask[h, w]
                        pred_class = pred_mask[h, w]
                        if true_class != pred_class:
                            confusion_matrix[true_class, pred_class] += 1
    
    # Анализ путаницы классов
    confusion_df = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    
    # Находим самые частые ошибки
    confusion_no_diag = confusion_matrix.copy()
    np.fill_diagonal(confusion_no_diag, 0)
    most_confused = np.unravel_index(np.argsort(confusion_no_diag, axis=None)[-10:], confusion_no_diag.shape)
    
    return {
        'boundary_error_rate': boundary_errors / max(total_boundary_pixels, 1),
        'small_object_error_rate': small_object_errors / max(total_small_objects, 1),
        'confusion_matrix': confusion_df,
        'most_confused_pairs': [(class_names[most_confused[0][i]], class_names[most_confused[1][i]], 
                                 confusion_no_diag[most_confused[0][i], most_confused[1][i]]) 
                                for i in range(len(most_confused[0]))]
    }

# ========== Запуск анализа ==========
print("\n" + "="*60)
print("АНАЛИЗ ОШИБОК МОДЕЛИ")
print("="*60)

# 1. Анализ ошибок по классам
print("\n1. Анализ ошибок по классам...")
class_ious, class_error_rates = analyze_class_errors(model, test_loader, device, NUM_CLASSES, class_names)

print(f"\n{'Класс':<25} {'IoU':<10} {'Error Rate':<12}")
print("-"*47)
for i, class_name in enumerate(class_names):
    if i < len(class_ious):
        print(f"{class_name:<25} {class_ious[i]:<10.4f} {class_error_rates[i]:<12.4f}")
print("-"*47)

# 2. Визуализация худших примеров
print("\n2. Визуализация изображений с наибольшими ошибками...")
worst_samples = visualize_errors(model, test_loader, device, num_samples=6)

# 3. Детальный анализ типов ошибок
print("\n3. Анализ типов ошибок...")
error_types = analyze_error_types(model, test_loader, device, class_names)

print(f"\nОшибки на границах объектов:")
print(f"  Доля ошибочных пикселей на границах: {error_types['boundary_error_rate']*100:.2f}%")

print(f"\nОшибки на мелких объектах:")
print(f"  Доля неправильно предсказанных мелких объектов: {error_types['small_object_error_rate']*100:.2f}%")

print(f"\nНаиболее частые путаницы классов:")
for true_class, pred_class, count in error_types['most_confused_pairs'][:5]:
    print(f"  {true_class} → {pred_class}: {count:.0f} пикселей")

# Сохраняем матрицу ошибок
error_types['confusion_matrix'].to_csv('confusion_matrix.csv', encoding='utf-8-sig')
print("\nМатрица ошибок сохранена в 'confusion_matrix.csv'")

print("\nАнализ ошибок завершен!")
print("   Результаты сохранены:")
print("   - error_analysis_worst_samples.png")
print("   - confusion_matrix.csv")