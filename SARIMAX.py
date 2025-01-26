import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# Подготовка данных
df = pd.read_csv("transport_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Агрегируем данные по дням
daily_hours = df.groupby('date')['hours'].sum()

# Разделение на тренировочную и тестовую выборки
train = daily_hours['2019-01-28':'2024-01-27']
test = daily_hours['2024-01-28':'2025-01-25']

# Обучение модели SARIMAX
start_time_train = time.time()
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 365), enforce_stationarity=False, enforce_invertibility=False)
sarimax_fit = model.fit(disp=False)
end_time_train = time.time()

# Прогноз
start_time_pred = time.time()
forecast = sarimax_fit.get_forecast(steps=len(test))
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()
end_time_pred = time.time()
# Оценка прогноза
mae = mean_absolute_error(test, forecast_values)
rmse = np.sqrt(mean_squared_error(test, forecast_values))
mape = np.mean(np.abs((test - forecast_values) / test)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Время обучения модели: {end_time_train - start_time_train:.2f} секунд")
print(f"Время предсказания модели: {end_time_pred - start_time_pred:.2f} секунд")

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(daily_hours.index, daily_hours, label="Фактические данные")
plt.plot(test.index, forecast_values, label="Прогноз (SARIMAX)", linestyle="--")
plt.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='gray', alpha=0.3, label="Доверительный интервал")
plt.xlabel("Дата")
plt.ylabel("Часы")
plt.title("Прогноз часов работы техники с помощью SARIMAX")
plt.legend()
plt.grid()
plt.show()

# Сохранение прогноза в CSV
forecast_df = pd.DataFrame({
    "date": test.index,
    "actual": test.values,
    "forecast": forecast_values.values,
    "lower_bound": forecast_ci.iloc[:, 0].values,
    "upper_bound": forecast_ci.iloc[:, 1].values
})
# forecast_df.to_csv("sarimax_forecast.csv", index=False)
# print("Прогноз сохранен в файл sarimax_forecast.csv")
