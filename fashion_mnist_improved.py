import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import functions as f
import matplotlib.pyplot as plt

RANDOM_SEED = 9
BATCH_SIZE = 64
NUM_CLASSES = 10
FIRST_LAYER = 32
SECOND_LAYER = 64
THIRD_LAYER = 128

# Стандартные статистики для FashionMNIST
mean = 0.5
std = 0.5

# ТРАНСФОРМАЦИЯ И АУГМЕНТАЦИЯ
train_transform = transforms.Compose([
    transforms.Pad(6),
    transforms.RandomCrop(28),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)), 
    transforms.Normalize((mean,), (std,))
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Загружаем данные
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=train_transform
)

f.visualize_augmentations(train_dataset, num_images=3, num_variations=4)

val_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=eval_transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=eval_transform
)

# Разделяем на train/val
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

indices = list(range(len(train_dataset)))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_subset = Subset(train_dataset, train_indices)  # с аугментацией
val_subset = Subset(val_dataset, val_indices)        # без аугментации


# DataLoaders
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class ImprovedFashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # извлекаем признаки
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, FIRST_LAYER, 3, padding=1),    
            nn.BatchNorm2d(FIRST_LAYER),                 
            nn.ReLU(),       
            nn.Conv2d(FIRST_LAYER, FIRST_LAYER, 3, padding=1),    
            nn.BatchNorm2d(FIRST_LAYER),                 
            nn.ReLU(),                       
            nn.MaxPool2d(2),    # 28x28 → 14x14            
            nn.Dropout2d(0.25),                   
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(FIRST_LAYER, SECOND_LAYER, 3, padding=1),    
            nn.BatchNorm2d(SECOND_LAYER),                 
            nn.ReLU(),       
            nn.Conv2d(SECOND_LAYER, SECOND_LAYER, 3, padding=1),    
            nn.BatchNorm2d(SECOND_LAYER),                 
            nn.ReLU(),                       
            nn.MaxPool2d(2),    # 14x14 -> 7           
            nn.Dropout2d(0.25),                   
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(SECOND_LAYER, THIRD_LAYER, 3, padding=1),    
            nn.BatchNorm2d(THIRD_LAYER),                 
            nn.ReLU(),       
            nn.Conv2d(THIRD_LAYER, THIRD_LAYER, 3, padding=1),    
            nn.BatchNorm2d(THIRD_LAYER),                 
            nn.ReLU(),                       
            nn.MaxPool2d(2),    # 7->3           
            nn.Dropout2d(0.25),                   
        )
        
        # принимаем решение
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Dropout(0.5),
            nn.Linear(THIRD_LAYER * 3 * 3, SECOND_LAYER),
            nn.ReLU(),                             
            nn.Dropout(0.3),                       
            nn.Linear(SECOND_LAYER, NUM_CLASSES)                     
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedFashionMNISTNet().to(device)


loss_fn = nn.CrossEntropyLoss()
# оптимизаторы
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Планировщик learning rate (уменьшает lr, когда точность перестает расти)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)


# Списки для сохранения метрик
train_losses = []
train_accuracies = []
val_accuracies = []
best_val_acc = 0
patience = 7
patience_counter = 0

# ОБУЧЕНИЕ 
def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Считаем accuracy во время обучения
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy 

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


if __name__ == "__main__":
    # ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
    print("Начинаем обучение...\n")
    num_epochs = 70

    for epoch in range(1, num_epochs + 1):
        # Обучение - теперь получаем и loss, и accuracy
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)  # Сохраняем train accuracy
        
        # Валидация
        val_acc = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)
        
        # Выводим и train loss, и train accuracy, и val accuracy
        print(f"Эпоха {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Сохраняем лучшую модель
            torch.save(model.state_dict(), 'best_model_improved.pth')
            print(f"  ✓ Новая лучшая модель! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\nРанняя остановка на эпохе {epoch}")
            break
        
        print("-" * 50)

    # Финальное тестирование
    model.load_state_dict(torch.load('best_model_improved.pth', weights_only=True))
    test_acc = evaluate(model, test_loader, device)

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ УЛУЧШЕННОЙ МОДЕЛИ:")
    print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
    print(f"Тестовая точность: {test_acc:.2f}%")
    print("="*60)

    # Создание графиков
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График функции потерь
    axes[0].plot(train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2)
    axes[0].set_xlabel('Эпоха', fontsize=12)
    axes[0].set_ylabel('Потери', fontsize=12)
    axes[0].set_title('Изменение функции потерь', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # График точности
    axes[1].plot(val_accuracies, label='Val Accuracy', marker='s', markersize=3, linewidth=2, color='red')
    axes[1].set_xlabel('Эпоха', fontsize=12)
    axes[1].set_ylabel('Точность (%)', fontsize=12)
    axes[1].set_title('Изменение точности', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


    # ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ УЛУЧШЕННОЙ МОДЕЛИ ====================
    next_num = f.get_next_experiment_number()

    # Описание улучшенной архитектуры
    architecture_desc = "Conv2d(1→32->32->62->64->128->128) + MaxPool(28->14->7->3) + Linear(128*3*3→64)→10"

    # Сохраняем результаты
    f.save_experiment_results(
        experiment_num=next_num + 1,
        description="Регуляризация 0.001",
        architecture_desc=architecture_desc,
        optimizer_name="AdamW",
        learning_rate=0.001,  # начальный LR
        num_epochs=epoch,
        val_accuracy=best_val_acc,
        test_accuracy=None,
        conclusion="",  # заполните потом в Excel
        filename="experiments.csv"
    )

    print("\n📊 Таблица экспериментов сохранена! Откройте experiments.csv в Excel")
    print("   и допишите выводы в последнюю колонку.")