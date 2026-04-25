import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np
import functions as f


RANDOM_SEED = 9
BATCH_SIZE = 64
NUM_CLASSES = 10
FIRST_LAYER = 32
SECOND_LAYER = 64

torch.manual_seed(RANDOM_SEED)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# РАЗДЕЛЕНИЕ ДАННЫХ
# Загружаем тренировочную часть датасета
full_train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Загружаем тестовую часть
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Разделяем тренировочную часть на train и validation
train_size = int(0.8 * len(full_train_dataset))  
val_size = len(full_train_dataset) - train_size  

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

print(f"Размер обучающей выборки: {len(train_dataset)}")
print(f"Размер валидационной выборки: {len(val_dataset)}")
print(f"Размер тестовой выборки: {len(test_dataset)}")

# СОЗДАНИЕ DATALOADER
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDataLoaders созданы:")
print(f"train_loader: {len(train_loader)} батчей")
print(f"val_loader: {len(val_loader)} батчей")
print(f"test_loader: {len(test_loader)} батчей")


# СОЗДАНИЕ СВЁРТОЧНОЙ НЕЙРОННОЙ СЕТИ
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # извлекаем признаки
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, FIRST_LAYER, 3, padding=1),    
            nn.BatchNorm2d(FIRST_LAYER),                 
            nn.ReLU(),                           
            nn.MaxPool2d(2),    # 28x28 → 14x14            
            nn.Dropout2d(0.25),                   
        )
        
        # принимаем решение
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Dropout(0.5),
            nn.Linear(FIRST_LAYER * 14 * 14, SECOND_LAYER),
            nn.ReLU(),                           
            nn.Dropout(0.3),                       
            nn.Linear(SECOND_LAYER, NUM_CLASSES)                     
        )

    def forward(self, x):
        x = self.conv_block(x) 
        x = self.classifier(x) 
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}\n")

model = FashionMNISTNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Списки для сохранения метрик
train_losses = []
val_accuracies = []
best_val_acc = 0
patience = 5
patience_counter = 0

# Функция для обучения одной эпохи
def train_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# валидация и тестирование
def evaluate(model, data_loader, device, return_predictions=False):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if return_predictions:
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    if return_predictions:
        return accuracy, np.array(all_predictions), np.array(all_targets)
    else:
        return accuracy

# ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
print("Начинаем обучение...\n")
num_epochs = 30

for epoch in range(1, num_epochs + 1):
    # Обучение
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    train_losses.append(train_loss)
    
    # Валидация
    val_acc = evaluate(model, val_loader, device, return_predictions=False)
    val_accuracies.append(val_acc)
    
    print(f"Эпоха {epoch}/{num_epochs}:")
    print(f"  Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Сохраняем лучшую модель
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  ✓ Новая лучшая модель! Val Acc: {val_acc:.2f}%")
    else:
        patience_counter += 1
        print(f"  Patience: {patience_counter}/{patience}")
        
    if patience_counter >= patience:
        print(f"\nРанняя остановка на эпохе {epoch}")
        break
    
    print("-" * 50)



# Загружаем лучшую модель для тестирования
print("\nЗагружаем лучшую модель для тестирования...")
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

# Финальная оценка на тестовом наборе (с возвратом предсказаний)
test_acc, test_predictions, test_targets = evaluate(
    model, test_loader, device, return_predictions=True
)
print(f"\n{'='*50}")
print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОМ НАБОРЕ:")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"{'='*50}")


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

# ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ BASELINE ====================
# Получаем следующий номер
next_num = f.get_next_experiment_number()

# Описание архитектуры для baseline
architecture_desc = "Conv2d(1→32) + MaxPool(28->14) + Linear(32*14*14→64)→10"

# Сохраняем результаты
f.save_experiment_results(
    experiment_num=next_num + 1,
    description="Baseline",
    architecture_desc=architecture_desc,
    optimizer_name="Adam",
    learning_rate=0.001,
    num_epochs=num_epochs, 
    val_accuracy=best_val_acc,
    test_accuracy=test_acc, 
    conclusion="",  # оставляем пустым для ручного заполнения
    filename="experiments.csv"
)
