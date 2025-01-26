import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import time

# Загрузка данных
df = pd.read_csv("transport_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Агрегируем данные по дням
daily_hours = df.groupby('date')['hours'].sum()

# Разделение на тренировочную и тестовую выборки
train = daily_hours['2019-01-28':'2024-01-27']
test = daily_hours['2024-01-28':'2025-01-25']

# Масштабирование данных
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Создание набора данных для XGBoost
def create_dataset_xgb(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 30  # Количество дней для анализа
X_train, y_train = create_dataset_xgb(train_scaled, look_back)
X_test, y_test = create_dataset_xgb(test_scaled, look_back)

# Преобразование для XGBoost
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Создание модели XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)

# Обучение модели
start_time_train = time.time()
model_xgb.fit(X_train, y_train)
end_time_train = time.time()

# Прогноз
start_time_pred = time.time()
train_predict_xgb = model_xgb.predict(X_train)
test_predict_xgb = model_xgb.predict(X_test)
end_time_pred = time.time()

# Обратное преобразование значений
train_predict_xgb = scaler.inverse_transform(train_predict_xgb.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict_xgb = scaler.inverse_transform(test_predict_xgb.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(daily_hours.index, daily_hours, label="Фактические данные")
plt.plot(train.index[look_back:], train_predict_xgb.flatten(), label="Прогноз XGBoost (тренировка)", linestyle="--")
plt.plot(test.index[look_back:], test_predict_xgb.flatten(), label="Прогноз XGBoost (тест)", linestyle="--")
plt.xlabel("Дата")
plt.ylabel("Часы")
plt.title("Прогноз часов работы техники с помощью XGBoost")
plt.legend()
plt.grid()
plt.show()

# Оценка прогноза
mae_xgb = mean_absolute_error(y_test_actual.flatten(), test_predict_xgb.flatten())
rmse_xgb = np.sqrt(mean_squared_error(y_test_actual.flatten(), test_predict_xgb.flatten()))
mape_xgb = np.mean(np.abs((y_test_actual - test_predict_xgb) / y_test_actual)) * 100

print(f"MAE (XGBoost): {mae_xgb:.2f}")
print(f"RMSE (XGBoost): {rmse_xgb:.2f}")
print(f"MAPE (XGBoost): {mape_xgb:.2f}%")
print(f"Время обучения модели: {end_time_train - start_time_train:.2f} секунд")
print(f"Время предсказания модели: {end_time_pred - start_time_pred:.2f} секунд")
