import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from sklearn.metrics import (accuracy_score, f1_score)
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
df = pd.read_csv('D:\study\sem5\СтатОИВ\лр3\enhanced_health_data.csv')

# Преобразование типов данных
df['Gender'] = df['Gender'].astype('category')
df['Smoker'] = df['Smoker'].astype('category')
df['Diabetes'] = df['Diabetes'].astype('category')
df['Health'] = df['Health'].astype('category')

print("Типы данных в датасете:")
print(df.dtypes)
print(f"\nРазмер датасета: {df.shape}")

columns_to_drop = ['Name', 'PatientID']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Анализ целевой переменной
print("\nРаспределение целевой переменной Health:")
print(df['Health'].value_counts())
print(f"Уникальные значения Health: {df['Health'].unique()}")

# Разделение на обучающую и тестовую выборки
df_train, df_test = train_test_split(df, test_size=0.2, random_state=9, stratify=df['Health'])

print(f"\nРазмер обучающей выборки: {df_train.shape}")
print(f"Размер тестовой выборки: {df_test.shape}")

# КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
categorical_features = ['Smoker', 'Diabetes', 'Gender']

# One-Hot Encoding для категориальных признаков (кроме Health)
encoded_train_dfs = []
encoded_test_dfs = []

for feature in categorical_features:
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    train_encoded = encoder.fit_transform(df_train[[feature]])
    test_encoded = encoder.transform(df_test[[feature]])
    
    feature_names = encoder.get_feature_names_out([feature])
    encoded_train_df = pd.DataFrame(train_encoded, columns=feature_names, index=df_train.index)
    encoded_test_df = pd.DataFrame(test_encoded, columns=feature_names, index=df_test.index)
    
    encoded_train_dfs.append(encoded_train_df)
    encoded_test_dfs.append(encoded_test_df)

# Объединяем все закодированные признаки
df_train_encoded = pd.concat(encoded_train_dfs, axis=1)
df_test_encoded = pd.concat(encoded_test_dfs, axis=1)

# Удаляем исходные категориальные признаки (кроме Health)
df_train = df_train.drop(categorical_features, axis=1)
df_test = df_test.drop(categorical_features, axis=1)

# Добавляем закодированные категориальные признаки
df_train_processed = pd.concat([df_train, df_train_encoded], axis=1)
df_test_processed = pd.concat([df_test, df_test_encoded], axis=1)

# СТАНДАРТИЗАЦИЯ ЧИСЛОВЫХ ПРИЗНАКОВ
scaler_minmax = StandardScaler()
numeric_columns = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Height (cm)', 'Weight (kg)', 'BMI']
df_train_processed[numeric_columns] = scaler_minmax.fit_transform(df_train_processed[numeric_columns])
df_test_processed[numeric_columns] = scaler_minmax.transform(df_test_processed[numeric_columns])

# Сохраняем целевую переменную
y_train = df_train_processed['Health'].copy()
y_test = df_test_processed['Health'].copy()

# Удаляем целевую переменную из признаков
X_train = df_train_processed.drop('Health', axis=1)
X_test = df_test_processed.drop('Health', axis=1)

# Кодируем целевую переменную для ROC-AUC
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')



# ПРЕОБРАЗОВАНИЕ В ТЕНЗОРЫ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}\n")

# Конвертация numpy в тензоры и перенос на GPU
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

print(f"X_train_tensor: {X_train_tensor.shape}")
print(f"y_train_tensor: {y_train_tensor.shape}")
print(f"X_test_tensor: {X_test_tensor.shape}")
print(f"y_test_tensor: {y_test_tensor.shape}")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
            )

    def forward(self, x):
        logits = self.stack(x)
        return logits

model = Net().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Списки для сохранения метрик
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for i in range(60):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_tensor)
    
    with torch.no_grad():
        accuracy = accuracy_score(torch.argmax(output, dim=1).cpu(), y_train_encoded)

        output_test = model(X_test_tensor)
        loss_test = loss_fn(output_test, y_test_tensor) 
        accuracy_test = accuracy_score(torch.argmax(output_test, dim=1).cpu(), y_test_encoded) 

    loss.backward()
    optimizer.step()

    # Сохраняем метрики
    train_losses.append(loss.item())
    test_losses.append(loss_test.item())
    train_accuracies.append(accuracy.item())
    test_accuracies.append(accuracy_test.item())

    print(f"Эпоха {i}: loss={loss:.2f}, accuracy={accuracy}, loss_test={loss_test:.2f}, accuracy_test={accuracy_test}.")

print(f"Точность модели: {(accuracy_test*100):.0f}%\n")

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Создание графиков
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График функции потерь
axes[0].plot(range(1, 61), train_losses, label='Train Loss', marker='o', markersize=3, linewidth=2)
axes[0].plot(range(1, 61), test_losses, label='Test Loss', marker='s', markersize=3, linewidth=2)
axes[0].set_xlabel('Эпоха', fontsize=12)
axes[0].set_ylabel('Потери', fontsize=12)
axes[0].set_title('Изменение функции потерь', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# График точности
axes[1].plot(range(1, 61), train_accuracies, label='Train Accuracy', marker='o', markersize=3, linewidth=2, color='green')
axes[1].plot(range(1, 61), test_accuracies, label='Test Accuracy', marker='s', markersize=3, linewidth=2, color='red')
axes[1].set_xlabel('Эпоха', fontsize=12)
axes[1].set_ylabel('Точность (%)', fontsize=12)
axes[1].set_title('Изменение точности', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


results = {
    'train_acc': accuracy,
    'test_acc': accuracy_test,
    'train_f1': f1_score(torch.argmax(output, dim=1).cpu(), y_train_encoded, average='weighted', zero_division=0),
    'test_f1': f1_score(torch.argmax(output_test, dim=1).cpu(), y_test_encoded, average='weighted', zero_division=0)
}





with open('classical_results.json', 'r') as f:
    classical = json.load(f)

df = pd.DataFrame({
    'Метрика': ['Train Accuracy', 'Test Accuracy', 'Train F1', 'Test F1'],
    'Decision tree': [
        classical['train_acc'], classical['Accuracy'],
        classical['train_f1'], classical['F1-Score']
    ],
    'Нейросеть': [
        results['train_acc'], results['test_acc'],
        results['train_f1'], results['test_f1']
    ]
})

print(df.to_string(index=False))
df.to_csv('comparison.csv', index=False)