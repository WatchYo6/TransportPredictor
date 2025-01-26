from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Подготовка данных
df = pd.read_csv("transport_data.csv")
df['date'] = pd.to_datetime(df['date'])

# Агрегируем данные по дням
daily_hours = df.groupby('date')['hours'].sum().reset_index()
daily_hours.columns = ['ds', 'y']  # Требуемые названия для Prophet

# Разделение на тренировочную и тестовую выборки
train = daily_hours[(daily_hours['ds'] >= '2019-01-28') & (daily_hours['ds'] <= '2024-01-27')]
test = daily_hours[(daily_hours['ds'] >= '2024-01-28') & (daily_hours['ds'] <= '2025-01-25')]

# Создание и обучение модели Prophet
start_time_train = time.time()
model = Prophet()
model.fit(train)
end_time_train = time.time()

# Прогноз
start_time_pred = time.time()
future = model.make_future_dataframe(periods=len(test), freq='D')
forecast = model.predict(future)
end_time_pred = time.time()

# Оценка прогноза
forecast_test = forecast[-len(test):]  # Прогнозируемые значения для тестового набора
mae = mean_absolute_error(test['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
mape = np.mean(np.abs((test['y'] - forecast_test['yhat']) / test['y'])) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Время обучения модели: {end_time_train - start_time_train:.2f} секунд")
print(f"Время предсказания модели: {end_time_pred - start_time_pred:.2f} секунд")

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(daily_hours['ds'], daily_hours['y'], label="Фактические данные")
plt.plot(forecast['ds'], forecast['yhat'], label="Прогноз", linestyle="--")
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.3, label="Доверительный интервал")
plt.xlabel("Дата")
plt.ylabel("Часы")
plt.title("Прогноз часов работы техники с помощью Prophet")
plt.legend()
plt.grid()
plt.show()

# Сохранение прогноза в CSV
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("prophet_forecast.csv", index=False)
# print("Прогноз сохранен в файл prophet_forecast.csv")
