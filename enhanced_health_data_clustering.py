import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('enhanced_health_data.csv')

# 1. ПРЕДОБРАБОТКА ДАННЫХ
print("=== ПРЕДОБРАБОТКА ДАННЫХ ===")

# Преобразование типов данных
df['Gender'] = df['Gender'].astype('category')
df['Smoker'] = df['Smoker'].astype('category')
df['Diabetes'] = df['Diabetes'].astype('category')
df['Health'] = df['Health'].astype('category')

print("Типы данных в датасете:")
print(df.dtypes)
print(f"\nРазмер датасета: {df.shape}")

# Удаление ненужных столбцов
columns_to_drop = ['Name', 'PatientID']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Проверка пропущенных значений
print("\nПропущенные значения:")
print(df.isnull().sum())

print("\n=== ОБРАБОТКА ВЫБРОСОВ ===")
def handle_outliers_iqr_smart(df, columns, iqr_multiplier=1.5, method='flag', min_outliers_pct=0.1):
    df_clean = df.copy()
    added_flags = []
    
    for col in columns:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Определяем выбросы
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            n_outliers = outliers.sum()
            pct_outliers = n_outliers / len(df_clean) * 100
            
            print(f"{col}: {n_outliers} выбросов ({pct_outliers:.1f}%)")
            
            if method == 'cap':
                # Ограничиваем выбросы границами
                df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
                df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
                print(f"  Выбросы ограничены границами [{lower_bound:.1f}, {upper_bound:.1f}]")
                
            elif method == 'flag' and n_outliers > 0 and pct_outliers >= min_outliers_pct:
                # Добавляем флаг выброса только если есть выбросы
                df_clean[f'{col}_outlier'] = outliers.astype(int)
                added_flags.append(f'{col}_outlier')
                print(f"Добавлен признак '{col}_outlier'")
            elif method == 'flag':
                print(f"Пропущено (нет выбросов или меньше {min_outliers_pct}%)")
    
    if added_flags:
        print(f"\nДобавлены флаги выбросов для: {added_flags}")
    else:
        print(f"\nНе добавлено ни одного флага выбросов")
    
    return df_clean

# Медицински значимые признаки для обработки выбросов
medical_outlier_features = [
    'Age',              
    'Systolic BP',     
    'Diastolic BP',   
    'Cholesterol',     
    'BMI'              
]

# Используем умную обработку
df = handle_outliers_iqr_smart(df, medical_outlier_features, method='flag', min_outliers_pct=0.1)
# Сохраняем целевую переменную для последующей валидации
true_labels = df['Health'].copy()

# Удаляем целевую переменную для кластеризации
df_cluster = df.drop('Health', axis=1)

# Кодирование категориальных переменных
categorical_features = ['Smoker', 'Diabetes', 'Gender']
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df_cluster[categorical_features])
feature_names = encoder.get_feature_names_out(categorical_features)

# Создаем DataFrame с закодированными признаками
df_encoded = pd.DataFrame(encoded_features, columns=feature_names, index=df_cluster.index)
df_processed = pd.concat([df_cluster.drop(categorical_features, axis=1), df_encoded], axis=1)

# Масштабирование числовых признаков
numeric_columns = ['Age', 'Systolic BP', 'Diastolic BP', 'Cholesterol', 
                   'Height (cm)', 'Weight (kg)', 'BMI']
scaler = StandardScaler()
df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

# Флаги выбросов оставляем как есть (0 или 1)
print(f"\nРазмер данных после предобработки: {df_processed.shape}")
print("Столбцы в обработанных данных:")
print(df_processed.columns.tolist())
print("\nПервые 5 строк обработанных данных:")
print(df_processed.head())

# 3. ПОСТРОЕНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ КЛАСТЕРИЗАЦИИ

# Функция для визуализации кластеров
def plot_clusters(X, labels, title, method='PCA'):
    plt.figure(figsize=(15, 5))
    
    if method == 'PCA':
        # PCA проекция
        pca = PCA(n_components=2, random_state=42)
        X_proj = pca.fit_transform(X)
        x_label, y_label = 'PC1', 'PC2'
        # Выводим объясненную дисперсию
        print(f"{title} - PCA объясненная дисперсия: {pca.explained_variance_ratio_.round(3)}")
    else:
        # t-SNE проекция
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_proj = tsne.fit_transform(X)
        x_label, y_label = 't-SNE1', 't-SNE2'
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{title} - {method}')
    
    plt.subplot(1, 2, 2)
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Кластер')
    plt.ylabel('Количество точек')
    plt.title('Размеры кластеров')
    
    plt.tight_layout()
    plt.show()

# A. KMEANS CLUSTERING
print("\n" + "=" * 50)
print("KMEANS CLUSTERING")
print("=" * 50)

# Силуэтный анализ для определения оптимального количества кластеров
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_processed)

    # Безопасный расчет silhouette score
    if len(set(kmeans.labels_)) > 1:
        silhouette_scores.append(silhouette_score(df_processed, kmeans.labels_))
    else:
        silhouette_scores.append(-1)

# Визуализация метода локтя и силуэтного анализа
plt.figure(figsize=(15, 5))

plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Количество кластеров')
plt.ylabel('Silhouette Score')
plt.title('Силуэтный анализ для KMeans')
plt.grid(True)

plt.tight_layout()
plt.show()

# Выбираем оптимальное количество кластеров
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"Оптимальное количество кластеров (по силуэтному анализу): {optimal_k}")

# Обучаем KMeans с оптимальным количеством кластеров
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(df_processed)

print(f"Кластеры KMeans: {np.unique(kmeans_labels)}")
print(f"Размеры кластеров: {np.bincount(kmeans_labels)}")

# Визуализация результатов KMeans
plot_clusters(df_processed, kmeans_labels, 'KMeans Clustering')

# B. AGGLOMERATIVE CLUSTERING
print("\n" + "=" * 50)
print("AGGLOMERATIVE CLUSTERING")
print("=" * 50)

# Построение дендрограммы для определения количества кластеров
plt.figure(figsize=(15, 6))

# Используем подвыборку для дендрограммы (для скорости)
sample_indices = np.random.choice(len(df_processed), min(100, len(df_processed)), replace=False)
df_sample = df_processed.iloc[sample_indices]

# Строим дендрограмму с разными методами linkage
methods = ['ward', 'complete', 'average', 'single']
for i, method in enumerate(methods, 1):
    plt.subplot(1, 4, i)
    try:
        linked = linkage(df_sample, method=method)
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title(f'Дендрограмма ({method} linkage)')
        plt.xlabel('Индекс образца')
        plt.ylabel('Расстояние')
    except Exception as e:
        plt.text(0.5, 0.5, f'Ошибка:\n{str(e)}', ha='center', va='center')
        plt.title(f'Дендрограмма ({method} linkage)')

plt.tight_layout()
plt.show()

# Обучаем Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg_clustering.fit_predict(df_processed)

print(f"Кластеры Agglomerative: {np.unique(agg_labels)}")
print(f"Размеры кластеров: {np.bincount(agg_labels)}")

# Визуализация результатов Agglomerative Clustering
plot_clusters(df_processed, agg_labels, 'Agglomerative Clustering')

# C. DBSCAN CLUSTERING
print("\n" + "=" * 50)
print("DBSCAN CLUSTERING")
print("=" * 50)

# Построение графика k-ближайших соседей для определения eps
plt.figure(figsize=(12, 8))

for min_samples in [3, 5, 7]:
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(df_processed)
    distances, indices = neighbors_fit.kneighbors(df_processed)
    distances = np.sort(distances[:, -1], axis=0)
    
    plt.plot(distances, label=f'min_samples={min_samples}')

plt.xlabel('Точки данных (отсортированные)')
plt.ylabel('Расстояние до k-го соседа')
plt.title('K-Distance Graph для определения eps')
plt.legend()
plt.grid(True)
plt.show()

# Улучшенная функция для расчета метрик
def safe_cluster_metrics(X, labels):
    """расчет метрик кластеризации с обработкой особых случаев"""
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_percentage': (n_noise / len(labels)) * 100,
        'silhouette': -1,
        'davies_bouldin': float('inf'),
        'calinski_harabasz': -1
    }
    
    # Исключаем выбросы для расчета метрик
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]
    
    # Нужно хотя бы 2 кластера и 2 точки в каждом для большинства метрик
    if len(X_filtered) > 1 and len(set(labels_filtered)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(X_filtered, labels_filtered)
        except:
            metrics['silhouette'] = -1
        
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X_filtered, labels_filtered)
        except:
            metrics['davies_bouldin'] = float('inf')
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_filtered, labels_filtered)
        except:
            metrics['calinski_harabasz'] = -1
    
    return metrics

# Подбор параметров DBSCAN
dbscan_results = []
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [3, 5, 7, 10]

print("Подбор параметров DBSCAN:")
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(df_processed)
        
        metrics = safe_cluster_metrics(df_processed, dbscan_labels)
        
        result = {
            'eps': eps,
            'min_samples': min_samples,
            **metrics
        }
        dbscan_results.append(result)
        
        # Выводим только хорошие результаты
        if metrics['n_clusters'] >= 2 and metrics['noise_percentage'] < 30:
            print(f"eps={eps:.1f}, min_samples={min_samples}: "
                  f"clusters={metrics['n_clusters']}, noise={metrics['noise_percentage']:.1f}%, "
                  f"silhouette={metrics['silhouette']:.3f}")

# Выбираем лучшие параметры DBSCAN
dbscan_df = pd.DataFrame(dbscan_results)

# Фильтруем результаты: минимум 2 кластера, шума < 30%
filtered_dbscan = dbscan_df[
    (dbscan_df['n_clusters'] >= 2) & 
    (dbscan_df['noise_percentage'] < 30)
]

if len(filtered_dbscan) > 0:
    best_dbscan = filtered_dbscan.loc[filtered_dbscan['silhouette'].idxmax()]
else:
    # Если нет хороших результатов, берем с максимальным silhouette
    best_dbscan = dbscan_df.loc[dbscan_df['silhouette'].idxmax()]

print(f"\nЛучшие параметры DBSCAN:")
print(f"eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}")
print(f"Кластеры: {best_dbscan['n_clusters']}, Шум: {best_dbscan['noise_percentage']:.1f}%")
print(f"Silhouette: {best_dbscan['silhouette']:.3f}")

# Обучаем DBSCAN с лучшими параметрами
dbscan = DBSCAN(eps=best_dbscan['eps'], min_samples=int(best_dbscan['min_samples']))
dbscan_labels = dbscan.fit_predict(df_processed)

print(f"Фактические кластеры DBSCAN: {np.unique(dbscan_labels)}")
cluster_sizes = [np.sum(dbscan_labels == i) for i in np.unique(dbscan_labels)]
print(f"Размеры кластеров: {cluster_sizes}")

# Визуализация результатов DBSCAN
plot_clusters(df_processed, dbscan_labels, 'DBSCAN Clustering')

# 4. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ
print("\n" + "=" * 50)
print("ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("=" * 50)

# Функция для расчета всех метрик
def calculate_all_metrics(X, labels, true_labels=None, algorithm_name=""):
    metrics = {
        'Algorithm': algorithm_name,
        'Number of Clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'Silhouette Score': -1,
        'Davies-Bouldin Index': float('inf'),
        'Calinski-Harabasz Index': -1
    }
    
    # Безопасный расчет метрик
    safe_metrics = safe_cluster_metrics(X, labels)
    metrics.update({
        'Silhouette Score': safe_metrics['silhouette'],
        'Davies-Bouldin Index': safe_metrics['davies_bouldin'],
        'Calinski-Harabasz Index': safe_metrics['calinski_harabasz']
    })
    
    # Для DBSCAN добавляем информацию о выбросах
    if -1 in labels:
        metrics['Number of Noise Points'] = safe_metrics['n_noise']
        metrics['Noise Percentage'] = safe_metrics['noise_percentage']
    
    # Дополнительные метрики, если есть истинные метки
    if true_labels is not None:
        try:
            metrics['ARI'] = adjusted_rand_score(true_labels, labels)
            metrics['NMI'] = normalized_mutual_info_score(true_labels, labels)
        except:
            metrics['ARI'] = -1
            metrics['NMI'] = -1
    
    return metrics

# Расчет метрик для всех алгоритмов
results = []

# KMeans
kmeans_metrics = calculate_all_metrics(df_processed, kmeans_labels, true_labels, "KMeans")
results.append(kmeans_metrics)

# Agglomerative Clustering
agg_metrics = calculate_all_metrics(df_processed, agg_labels, true_labels, "Agglomerative")
results.append(agg_metrics)

# DBSCAN
dbscan_metrics = calculate_all_metrics(df_processed, dbscan_labels, true_labels, "DBSCAN")
results.append(dbscan_metrics)

# Создаем таблицу результатов
results_df = pd.DataFrame(results)

# Форматируем вывод для лучшей читаемости
display_columns = ['Algorithm', 'Number of Clusters', 'Silhouette Score', 
                   'Davies-Bouldin Index', 'Calinski-Harabasz Index']
if 'ARI' in results_df.columns:
    display_columns.extend(['ARI', 'NMI'])
if 'Number of Noise Points' in results_df.columns:
    display_columns.extend(['Number of Noise Points', 'Noise Percentage'])

print("\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ:")
print(results_df[display_columns].round(3).to_string(index=False))

# 5. АНАЛИЗ РЕЗУЛЬТАТОВ
print("\n" + "=" * 50)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Определяем лучший алгоритм по Silhouette Score (исключая невалидные значения)
valid_results = results_df[results_df['Silhouette Score'] > -1]
if len(valid_results) > 0:
    best_algorithm = valid_results.loc[valid_results['Silhouette Score'].idxmax(), 'Algorithm']
    best_score = valid_results['Silhouette Score'].max()
    print(f"Лучший алгоритм по Silhouette Score: {best_algorithm} ({best_score:.3f})")
else:
    best_algorithm = results_df.loc[results_df['Calinski-Harabasz Index'].idxmax(), 'Algorithm']
    print(f"Лучший алгоритм по Calinski-Harabasz Index: {best_algorithm}")

# Выбираем метки лучшего алгоритма для анализа
if best_algorithm == "KMeans":
    best_labels = kmeans_labels
elif best_algorithm == "Agglomerative":
    best_labels = agg_labels
else:
    best_labels = dbscan_labels

# Анализ: как выбросы распределены по кластерам
print("\nРАСПРЕДЕЛЕНИЕ ВЫБРОСОВ ПО КЛАСТЕРАМ:")
cluster_stats = pd.DataFrame({
    'Cluster': best_labels,
    'Diastolic BP_outlier': df_processed['Diastolic BP_outlier'] if 'Diastolic BP_outlier' in df_processed.columns else 0,
    'BMI_outlier': df_processed['BMI_outlier'] if 'BMI_outlier' in df_processed.columns else 0,
    'Systolic BP_outlier': df_processed['Systolic BP_outlier'] if 'Systolic BP_outlier' in df_processed.columns else 0,
    'Cholesterol_outlier': df_processed['Cholesterol_outlier'] if 'Cholesterol_outlier' in df_processed.columns else 0
})

# Исключаем шумовые точки (кластер -1) из анализа
if -1 in cluster_stats['Cluster'].values:
    cluster_stats = cluster_stats[cluster_stats['Cluster'] != -1]

if len(cluster_stats) > 0:
    for cluster in np.unique(cluster_stats['Cluster']):
        cluster_data = cluster_stats[cluster_stats['Cluster'] == cluster]
        print(f"\nКластер {cluster} (размер: {len(cluster_data)}):")
        
        # Процент выбросов в кластере
        outlier_cols = [col for col in cluster_stats.columns if col.endswith('_outlier')]
        for col in outlier_cols:
            if col in cluster_data.columns:
                outlier_pct = cluster_data[col].sum() / len(cluster_data) * 100
                if outlier_pct > 0:
                    print(f"  {col}: {outlier_pct:.1f}% ({cluster_data[col].sum()}/{len(cluster_data)})")

# Boxplot распределения признаков по кластерам (только для не-выбросов)
plt.figure(figsize=(16, 12))
mask = best_labels != -1 if -1 in best_labels else slice(None)
filtered_labels = best_labels[mask]
filtered_data = df_processed[mask]

unique_clusters = np.unique(filtered_labels)

if len(unique_clusters) > 1:
    for i, col in enumerate(numeric_columns[:6], 1):
        plt.subplot(2, 3, i)
        cluster_data = []
        
        for cluster in unique_clusters:
            cluster_data.append(filtered_data[filtered_labels == cluster][col])
        
        plt.boxplot(cluster_data, labels=[f'Cluster {c}' for c in unique_clusters])
        plt.title(f'Распределение {col} по кластерам')
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Анализ характеристик кластеров
    print("\nХАРАКТЕРИСТИКИ КЛАСТЕРОВ (для лучшего алгоритма):")
    cluster_stats_full = df_processed.copy()
    cluster_stats_full['Cluster'] = best_labels
    
    for cluster in np.unique(best_labels):
        if cluster != -1:  # Исключаем выбросы
            cluster_data = cluster_stats_full[cluster_stats_full['Cluster'] == cluster]
            print(f"\nКластер {cluster} (размер: {len(cluster_data)}):")
            print("Средние значения числовых признаков:")
            for col in numeric_columns[:4]:  # Показываем только первые 4 признака
                print(f"  {col}: {cluster_data[col].mean():.3f}")

# 6. ВЫВОДЫ
print("\n" + "=" * 50)
print("ВЫВОДЫ")
print("=" * 50)

print("1. СРАВНЕНИЕ АЛГОРИТМОВ:")
for _, row in results_df.iterrows():
    print(f"   - {row['Algorithm']}: Silhouette = {row['Silhouette Score']:.3f}, "
          f"Clusters = {row['Number of Clusters']}")

print(f"\n2. ЛУЧШИЙ АЛГОРИТМ: {best_algorithm}")

if 'ARI' in results_df.columns:
    best_ari = results_df['ARI'].max()
    best_nmi = results_df['NMI'].max()
    print(f"3. СОГЛАСОВАННОСТЬ С ИСТИННЫМИ МЕТКАМИ:")
    print(f"   - Лучший ARI: {best_ari:.3f}")
    print(f"   - Лучший NMI: {best_nmi:.3f}")
