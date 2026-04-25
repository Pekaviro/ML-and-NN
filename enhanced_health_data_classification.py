import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
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

# 2. ОБРАБОТКА ВЫБРОСОВ
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

numeric_columns = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 'Height (cm)', 'Weight (kg)', 'BMI']

for i, col in enumerate(numeric_columns):
    row = i // 4
    col_idx = i % 4 
    sns.boxplot(data=df, y=col, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Boxplot of {col}')
    axes[row, col_idx].set_ylabel(col)

if len(numeric_columns) < 8:
    axes[1, 3].set_visible(False)

plt.tight_layout()
plt.show()

def detect_outliers_iqr(series, iqr_multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    return ((series < lower_bound) | (series > upper_bound)).astype(int)

def create_test_outliers(train_series, test_series, iqr_multiplier=1.5):
    Q1 = train_series.quantile(0.25)
    Q3 = train_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    return ((test_series < lower_bound) | (test_series > upper_bound)).astype(int)

# Обработка выбросов для тренировочной выборки
df_train["Diastolic BP_outlier"] = detect_outliers_iqr(df_train["Diastolic BP"])
df_train["BMI_outlier"] = detect_outliers_iqr(df_train["BMI"])

# Обработка выбросов для тестовой выборки
df_test["Diastolic BP_outlier"] = create_test_outliers(df_train["Diastolic BP"], df_test["Diastolic BP"])
df_test["BMI_outlier"] = create_test_outliers(df_train["BMI"], df_test["BMI"])

# 3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
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

# 4. МАСШТАБИРОВАНИЕ ПРИЗНАКОВ
scaler_minmax = MinMaxScaler()
df_train_processed[numeric_columns] = scaler_minmax.fit_transform(df_train_processed[numeric_columns])
df_test_processed[numeric_columns] = scaler_minmax.transform(df_test_processed[numeric_columns])

# 5. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИРОВАНИЯ
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

print(f"\nПодготовка данных завершена:")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train_encoded.shape}, y_test: {y_test_encoded.shape}")
print(f"Классы целевой переменной: {le.classes_}")

# 6. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=9, max_depth=5),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=9, max_iter=1000),
    'Naive Bayes': GaussianNB()
}

# Словари для хранения результатов
results = {}
predictions = {}
probabilities = {}

print("\n" + "="*50)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("="*50)

for name, model in models.items():
    print(f"\nОбучение модели: {name}")
    
    # Обучение модели
    model.fit(X_train, y_train_encoded)
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    y_train_pred = model.predict(X_train)
    
    # Сохраняем результаты
    predictions[name] = y_pred
    probabilities[name] = y_pred_proba
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        
        # для сравнения с нейросетью
        'train_acc': accuracy_score(y_train_encoded, y_train_pred),
        'train_f1': f1_score(y_train_encoded, y_train_pred, average='weighted', zero_division=0)
    }
    
    print(f"✓ {name} обучена")
    print(f"  Accuracy: {accuracy:.4f}")

# 7. СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
print("\n" + "="*60)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print(results_df)

# Визуализация сравнительной таблицы
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x_pos = np.arange(len(models))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, metric in enumerate(metrics_to_plot):
    axes[i].bar(x_pos, results_df[metric], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[i].set_title(f'Сравнение {metric}')
    axes[i].set_xticks(x_pos)
    axes[i].set_xticklabels(results_df.index, rotation=45)
    axes[i].set_ylabel(metric)
    
    # Добавляем значения на столбцы
    for j, v in enumerate(results_df[metric]):
        axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Общая сравнительная диаграмма
axes[5].bar(x_pos - 0.3, results_df['Accuracy'], width=0.2, label='Accuracy', color='blue')
axes[5].bar(x_pos - 0.1, results_df['Precision'], width=0.2, label='Precision', color='orange')
axes[5].bar(x_pos + 0.1, results_df['Recall'], width=0.2, label='Recall', color='green')
axes[5].bar(x_pos + 0.3, results_df['F1-Score'], width=0.2, label='F1-Score', color='red')
axes[5].set_title('Сравнение всех метрик по моделям')
axes[5].set_xticks(x_pos)
axes[5].set_xticklabels(results_df.index, rotation=45)
axes[5].set_ylabel('Score')
axes[5].legend()

plt.tight_layout()
plt.show()

# 8. MATRICES ОШИБОК (CONFUSION MATRIX)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

class_names = le.classes_

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(f'Confusion Matrix - {name}')
    axes[idx].set_xlabel('Предсказанные значения')
    axes[idx].set_ylabel('Истинные значения')

plt.tight_layout()
plt.show()

# 11. ВЫВОД ЛУЧШЕЙ МОДЕЛИ
best_model = results_df['Accuracy'].idxmax()
best_accuracy = results_df.loc[best_model, 'Accuracy']

print("\n" + "="*50)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("="*50)
print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
print(f"🎯 Точность лучшей модели: {best_accuracy:.4f}")


# Сохраняем результаты
import json
with open('classical_results.json', 'w') as f:
    json.dump(results[best_model], f)