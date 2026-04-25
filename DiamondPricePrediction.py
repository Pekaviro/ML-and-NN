import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('Diamond Price Prediction.csv')

df = df.rename(columns = {
    'Carat(Weight of Daimond)': 'Carat',
    'Cut(Quality)': 'Cut', 
    'Price(in US dollars)': 'Price',
    'X(length)': 'X',
    'Y(width)': 'Y', 
    'Z(Depth)': 'Z'
})

df['Cut'] = df['Cut'].astype('category')
df['Color'] = df['Color'].astype('category')
df['Clarity'] = df['Clarity'].astype('category')

# 2. Предобработка данных
# Разделение на обучающую и тестовую выборки
df_train, df_test = train_test_split(df, test_size=0.3, random_state=9)

# Анализ пропущенных значений и их обработка
print("\nКоличество пропусков:\n", df.isnull().sum())

# Обнаружение и обработка выбросов 
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

numerical_cols = ['Carat', 'Price', 'Table', 'Depth', 'X', 'Y', 'Z']

for i, col in enumerate(numerical_cols):
    row = i // 4
    col_idx = i % 4 
    sns.boxplot(data=df, y=col, ax=axes[row, col_idx])
    axes[row, col_idx].set_title(f'Boxplot of {col}')
    axes[row, col_idx].set_ylabel(col)

if len(numerical_cols) < 8:
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

df_train["Carat_outlier"] = detect_outliers_iqr(df_train["Carat"])
df_train["Depth_outlier"] = detect_outliers_iqr(df_train["Depth"])
df_train["Table_outlier"] = detect_outliers_iqr(df_train["Table"])
df_train["X_outlier"] = detect_outliers_iqr(df_train["X"])
df_train["Y_outlier"] = detect_outliers_iqr(df_train["Y"])
df_train["Z_outlier"] = detect_outliers_iqr(df_train["Z"])

# Создаем выбросы для тестовой выборки с теми же границами
def create_test_outliers(train_series, test_series, iqr_multiplier=1.5):
    Q1 = train_series.quantile(0.25)
    Q3 = train_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    return ((test_series < lower_bound) | (test_series > upper_bound)).astype(int)

df_test["Carat_outlier"] = create_test_outliers(df_train["Carat"], df_test["Carat"])
df_test["Depth_outlier"] = create_test_outliers(df_train["Depth"], df_test["Depth"])
df_test["Table_outlier"] = create_test_outliers(df_train["Table"], df_test["Table"])
df_test["X_outlier"] = create_test_outliers(df_train["X"], df_test["X"])
df_test["Y_outlier"] = create_test_outliers(df_train["Y"], df_test["Y"])
df_test["Z_outlier"] = create_test_outliers(df_train["Z"], df_test["Z"])

# Кодирование категориальных переменных 
cut_categories = [["Fair", "Good", "Very Good", "Premium", "Ideal"]]

encoder = OrdinalEncoder(categories=cut_categories)
df_train["Cut"] = encoder.fit_transform(df_train[["Cut"]])
df_test["Cut"] = encoder.transform(df_test[["Cut"]])


color_categories = [["J", "I", "H", "G", "F", "E", "D"]]

encoder = OrdinalEncoder(categories=color_categories)
df_train["Color"] = encoder.fit_transform(df_train[["Color"]])
df_test["Color"] = encoder.transform(df_test[["Color"]])


clarity_categories = [["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]]

encoder = OrdinalEncoder(categories=clarity_categories)
df_train["Clarity"] = encoder.fit_transform(df_train[["Clarity"]])
df_test["Clarity"] = encoder.transform(df_test[["Clarity"]])



# Нормализация/стандартизация признаков
scalers = {}
scaled_columns = []
for col in ["Carat", "Depth", "Table", "X", "Y", "Z"]:
    # Создаем и обучаем скейлер на тренировочных данных
    scaler = RobustScaler()
    df_train[f"{col}_scaled"] = scaler.fit_transform(df_train[[col]])
    
    # Сохраняем скейлер для тестовой выборки
    scalers[col] = scaler
    scaled_columns.append(f"{col}_scaled")

# Применяем к тестовой выборке
for col in ["Carat", "Depth", "Table", "X", "Y", "Z"]:
    df_test[f"{col}_scaled"] = scalers[col].transform(df_test[[col]])

print(df_train.head())


# 3. Построение и обучение моделей
X = df_train.drop('Price', axis=1)
y = df_train['Price']

X_test = df_test.drop('Price', axis=1)
y_test = df_test['Price']

# Линейная регрессия 
linear_regression = LinearRegression()
linear_model = linear_regression.fit(X, y)

score = linear_model.score(X_test, y_test)
print(f"Линейная регрессия R²: {score:.3f}")


# Полиномиальная регрессия степени
# Выделяем только числовые признаки для полиномиального преобразования
numeric_features = ['Carat', 'Depth', 'Table', 'X', 'Y', 'Z']
categorical_features = ['Cut', 'Color', 'Clarity']

X_numeric = df_train[numeric_features]
X_categorical = df_train[categorical_features]

X_test_numeric = df_test[numeric_features]
X_test_categorical = df_test[categorical_features]

# Полиномиальное преобразование только категориальных признаков
poly_categorical = PolynomialFeatures(degree=2, include_bias=False)
X_poly_categorical = poly_categorical.fit_transform(X_categorical)
X_test_poly_categorical = poly_categorical.transform(X_test_categorical)

# Объединяем с числовыми признаками
X_combined = np.hstack([X_poly_categorical, X_numeric.values])
X_test_combined = np.hstack([X_test_poly_categorical, X_test_numeric.values])

model_poly_combined = LinearRegression()
model_poly_combined.fit(X_combined, y)
score_poly_combined = model_poly_combined.score(X_test_combined, y_test)
print(f"Полиномиальная регрессия степени R² (категориальные полиномы + числовые значения): {score_poly_combined:.3f}")


# Регрессия с регуляризацией Ridge
ridge_regression = RidgeCV()
ridge_model = ridge_regression.fit(X, y)

score = ridge_model.score(X_test, y_test)
print(f"Ridge R²: {score:.3f}")


# Регрессия с регуляризацией Lasso
lasso_regression = LassoCV(max_iter=10000)
lasso_model = lasso_regression.fit(X, y)

score = lasso_model.score(X_test, y_test)
print(f"Lasso R²: {score:.3f}")


# Случайный лес 
forest_regression = RandomForestRegressor(n_estimators=6, random_state=9)
forest_model = forest_regression.fit(X, y)

score = forest_model.score(X_test, y_test)
print(f"Случайный лес R²: {score:.3f}")


# Градиентный бустинг
gradient_regression = GradientBoostingRegressor(n_estimators=100, max_depth=5)
gradient_model = gradient_regression.fit(X, y)

score = gradient_model.score(X_test, y_test)
print(f"Градиентный бустинг R²: {score:.3f}")


# CatBoost
df_train_cat, df_test_cat = train_test_split(df, test_size=0.3, random_state=9)

X_cat = df_train_cat.drop('Price', axis=1)
y_cat = df_train_cat['Price']
X_test_cat = df_test_cat.drop('Price', axis=1)
y_test_cat = df_test_cat['Price']

cat_regression = CatBoostRegressor(cat_features=categorical_features, early_stopping_rounds=50, random_state=9, verbose=False)

# Обучаем модель
cat_model = cat_regression.fit(
    X_cat, y_cat,
    eval_set=(X_test_cat, y_test_cat)
)

score = cat_model.score(X_test_cat, y_test_cat)
print(f"CatBoost R²: {score:.3f}")


# 4. Оценка качества моделей
def calculate_metrics(model, X, y_true, model_name):
    # Предсказания
    y_pred = model.predict(X)
    
    # Вычисление метрик
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'R²': round(r2, 4)
    }

# Список для хранения результатов
results = []
results.append(calculate_metrics(linear_model, X_test, y_test, "Linear Regression"))
results.append(calculate_metrics(model_poly_combined, X_test_combined, y_test, "Polynomial Regression"))
results.append(calculate_metrics(ridge_model, X_test, y_test, "Ridge Regression"))
results.append(calculate_metrics(lasso_model, X_test, y_test, "Lasso Regression"))
results.append(calculate_metrics(forest_model, X_test, y_test, "Random Forest"))
results.append(calculate_metrics(gradient_model, X_test, y_test, "Gradient Boosting"))
results.append(calculate_metrics(cat_model, X_test_cat, y_test_cat, "CatBoost"))

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ РЕГРЕССИИ")
print("="*80)
formatted_table = results_df.to_string(
    index=False,
    justify='center', 
    col_space=15,     
    float_format='{:,.3f}'.format
)
print(formatted_table)


# 5. Выбор лучшей модели и анализ результатов
print("\nПо таблице можно заметить, что самые низкие показатели MAE, MSE и RMSE имеет CatBoost модель. Самые большие значения R² принадлежат Gradient Boosting и CatBoost. Можно сделать вывод, что лучшей из предложенных моделью является CatBoost.")

print("\n" + "="*80)
print("АНАЛИЗ МОДЕЛИ CatBoost")
print("="*80)

# Анализ важности признаков
feature_importance = cat_model.get_feature_importance()
feature_names = X_cat.columns

# Создаем DataFrame с важностью признаков
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nВАЖНОСТЬ ПРИЗНАКОВ:")
print(importance_df.to_string(index=False, justify='center'))

# Визуализация важности признаков
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Feature', palette='viridis')
plt.title('Важность признаков в модели CatBoost', fontsize=16, fontweight='bold')
plt.xlabel('Важность', fontsize=12)
plt.ylabel('Признаки', fontsize=12)
plt.tight_layout()
plt.show()

# Предсказания на тестовой выборке
y_pred_cat = cat_model.predict(X_test_cat)

# Создаем DataFrame с реальными и предсказанными значениями
comparison_df = pd.DataFrame({
    'Actual_Price': y_test_cat.values,
    'Predicted_Price': y_pred_cat,
    'Absolute_Error': np.abs(y_test_cat.values - y_pred_cat),
    'Relative_Error': np.abs((y_test_cat.values - y_pred_cat) / y_test_cat.values) * 100
})

# Добавляем исходные признаки для анализа
comparison_df = pd.concat([comparison_df, X_test_cat.reset_index(drop=True)], axis=1)

# 1. График предсказаний vs фактические значения
plt.figure(figsize=(12, 6))

plt.scatter(comparison_df['Actual_Price'], comparison_df['Predicted_Price'], 
           alpha=0.6, color='blue', s=20)
plt.plot([comparison_df['Actual_Price'].min(), comparison_df['Actual_Price'].max()],
         [comparison_df['Actual_Price'].min(), comparison_df['Actual_Price'].max()], 
         'r--', linewidth=2, label='Идеальная линия')
plt.xlabel('Фактическая цена ($)')
plt.ylabel('Предсказанная цена ($)')
plt.title('Фактические vs Предсказанные значения\n(CatBoost)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("ВЫВОДЫ ПО АНАЛИЗУ CATBOOST МОДЕЛИ:")
print("="*80)
print("1. Наиболее важные признаки для модели:", list(importance_df.head(3)['Feature']))
print(f"2. Модель в среднем ошибается на {comparison_df['Absolute_Error'].mean():.2f}$ ({comparison_df['Relative_Error'].mean():.2f}%)")