import pandas as pd
import numpy as np
from scipy.stats import pearsonr

df = pd.read_csv('crocodile_dataset.csv')


print(f"Размер датасета: {df.shape}\n")

print("Типы данных:\n", df.dtypes, "\n")

print("Первые строки:\n", df.head())
print("\nПоследние строки:\n", df.tail())

print("\nОбщая информация:\n")
print(df.info())
print(df.describe())



# Предобработка данных
print("\nКоличество пропусков:\n", df.isnull().sum())
df_cleaned = df.dropna()

print(f"Количество дубликатов: {df_cleaned.duplicated().sum()}")
df_unique = df_cleaned.drop_duplicates()

df_filtered = df.copy()

for column in df_filtered.select_dtypes(include=['number']).columns:
    # Метод межквартильного размаха (IQR)
    Q1 = df_filtered[column].quantile(0.25)
    Q3 = df_filtered[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Фильтрация выбросов
    df_filtered = df_filtered[(df_filtered[column] >= lower_bound) & (df_filtered[column] <= upper_bound)]

print("\nПосле обработки выбросов:")
print(f"Размер датасета: {df_filtered.shape}")
print(f"Количество строк: {len(df_filtered)}")
print(f"Удалено строк: {len(df_unique) - len(df_filtered)}")

df_filtered['Age Class'] = df_filtered['Age Class'].astype('category')
df_filtered['Sex'] = df_filtered['Sex'].astype('category')
df_filtered['Date of Observation'] = pd.to_datetime(df_filtered['Date of Observation'], format='%d-%m-%Y')

print("Новые типы данных:\n", df_filtered.dtypes, "\n")

# Кодирование категориальных переменных
df_encoded = pd.get_dummies(df_filtered, columns=['Age Class'])
df_encoded = pd.get_dummies(df_filtered, columns=['Sex'])

print('\n')


print(df_filtered[['Sex', 'Age Class']].iloc[1:6])
print('\n')


adult = df_filtered[df_filtered['Age Class']=='Adult']
print(adult.shape[0])

# 1. Какой вид крокодила самый популярный?
mode = df_encoded['Common Name'].mode()[0]
print(f"1. Какой вид крокодила самый популярный? - {mode}")

# 2. Какой род крокодилов самый редкий?
rarest = df_encoded['Genus'].value_counts().idxmin()
print(f"2. Какой род крокодилов самый редкий? - {rarest}")

# 3. Есть ли связь между длиной и весом крокодила?
df_numeric = df_encoded.select_dtypes(include=[np.number])
correlation_matrix = df_numeric.corr()

correlation = df_numeric['Observed Length (m)'].corr(df_numeric['Observed Weight (kg)'])
corr_coef, p_value = pearsonr(df_numeric['Observed Length (m)'], df_numeric['Observed Weight (kg)'])

if p_value < 0.05:
    if abs(corr_coef) > 0.7:
        answer = "Да, существует сильная статистически значимая связь."
    elif abs(corr_coef) > 0.3:
        answer = "Да, связь есть, но она умеренной силы."
    else:
        answer = "Связь статистически значима, но очень слабая."
else:
    answer = "Нет, статистически значимой связи не обнаружено."

print(f"3. Есть ли связь между длиной и весом крокодила? - {answer}")

# 4. Какой природоохранный статус встречается чаще?
mode = df_encoded['Conservation Status'].mode()[0]
print(f"4. Какой природоохранный статус встречается чаще? - {mode}")

# 5. Какой регион имеет наибольшее количество наблюдений?
mode = df_encoded['Country/Region'].mode()[0]
print(f"5. Какой регион имеет наибольшее количество наблюдений? - {mode}")

# 6. Какая среда обитания наиболее распространена?
mode = df_encoded['Habitat Type'].mode()[0]
print(f"6. Какая среда обитания наиболее распространена? - {mode}")

# 7. Какой вид чаще всего встречаются в любых реках?
river_data = df_encoded[df_encoded['Habitat Type'].str.contains('Rivers', case=False, na=False)]
mode = river_data['Scientific Name'].mode()[0]
print(f"7. Какой вид чаще всего встречаются в любых реках? - {mode}")

# 8. Какой возрастной класс самый редкий в выборке?
rarest = df_encoded['Age Class'].value_counts().idxmin()
print(f"8. Какой возрастной класс самый редкий в выборке? - {rarest}")

# 9. Какая страна/регион чаще всего фиксирует редкие виды на грани вымирания?
status_data = df_encoded[df_encoded['Conservation Status'] == 'Critically Endangered']
mode = status_data['Country/Region'].mode()[0]
print(f"9. Какая страна/регион чаще всего фиксирует редкие виды на грани вымирания? - {mode}")

# 10. Какая среда обитания характерна для самых тяжёлых крокодилов?
weight_by_habitat = df_encoded.groupby('Habitat Type').agg({
    'Observed Weight (kg)': ['mean', 'std', 'count', 'max']
}).round(1)
heaviest_habitat = weight_by_habitat[('Observed Weight (kg)', 'mean')].idxmax()
print(f"10. Какая среда обитания характерна для самых тяжёлых крокодилов? - {heaviest_habitat}")





import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Настройка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Гистограмма 1
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['Observed Length (m)'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Распределение длины крокодилов')
plt.xlabel('Длина')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

# Гистограмма 2
plt.figure(figsize=(10, 6))
plt.hist(df_filtered['Observed Weight (kg)'], bins=30, alpha=0.7, edgecolor='black')
plt.title('Распределение веса крокодилов')
plt.xlabel('Вес')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)
plt.show()

# Box Plot 1
plt.figure(figsize=(8, 6))
plt.boxplot(df_filtered['Observed Length (m)'])
plt.title('Коробчатая диаграмма длин')
plt.ylabel('Длина')

# Box Plot 2
plt.figure(figsize=(8, 6))
plt.boxplot(df_filtered['Observed Weight (kg)'])
plt.title('Коробчатая диаграмма весов')
plt.ylabel('Вес')
plt.show()

# Bar Plot 1
plt.figure(figsize=(10, 6))
df_filtered['Age Class'].value_counts().plot(kind='bar')
plt.title('Распределение по возрастному классу')
plt.xlabel('Возрастной класс')
plt.ylabel('Количество')
plt.xticks(rotation=45)


# Bar Plot 2
plt.figure(figsize=(10, 6))
df_filtered['Sex'].value_counts().plot(kind='bar')
plt.title('Распределение по полу')
plt.xlabel('Пол')
plt.ylabel('Количество')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix
corr_matrix = df_filtered.select_dtypes(include=[np.number]).corr()
# Тепловая карта корреляций
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
square=True, fmt='.2f', cbar_kws={'label': 'Коэффициент корреляции'})
plt.title('Матрица корреляций')
plt.show()

# Scatter plot 1
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Observed Length (m)'], df_filtered['Observed Weight (kg)'], alpha=0.6)
plt.xlabel('Длина')
plt.ylabel('Вес')
plt.title('Зависимость веса от длины')

# Scatter plot 2
plt.figure(figsize=(10, 6))
plt.scatter(df_filtered['Observation ID'], df_filtered['Observed Length (m)'], alpha=0.6)
plt.xlabel('ID')
plt.ylabel('Длина')
plt.title('Зависимость длины от ID :)))')
plt.show()

# Contingency table
contingency_table = pd.crosstab(df_filtered['Age Class'], df_filtered['Sex'])
plt.figure(figsize=(10, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues')
plt.title('Тепловая карта таблицы сопряженности')
plt.show()