import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Создание набора данных для LSTM
def create_dataset(data, look_back=30):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 30  # Количество предыдущих дней для анализа
X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

# Преобразование для LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Создание модели LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
# Время обучения
start_time_train = time.time()
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
end_time_train = time.time()

# Прогноз
start_time_pred = time.time()
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
end_time_pred = time.time()

# Обратное преобразование значений
train_predict = scaler.inverse_transform(train_predict)
y_train_actual = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test_actual = scaler.inverse_transform(y_test)

# Оценка прогноза
mae = mean_absolute_error(y_test_actual, test_predict)
rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
mape = np.mean(np.abs((y_test_actual - test_predict) / y_test_actual)) * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Время обучения модели: {end_time_train - start_time_train:.2f} секунд")
print(f"Время предсказания модели: {end_time_pred - start_time_pred:.2f} секунд")
# Сохранение прогноза в CSV файл
predictions_df = pd.DataFrame({
    'Date': test.index[look_back:],  # Даты для тестового набора
    'True Hours': y_test_actual.flatten(),
    'Predicted Hours': test_predict.flatten()
})

# predictions_df.to_csv('predictions.csv', index=False)
# print("Прогноз сохранен в файл predictions.csv")

# Визуализация
plt.figure(figsize=(12, 6))
plt.plot(daily_hours.index, daily_hours, label="Фактические данные")
plt.plot(train.index[look_back:], train_predict.flatten(), label="Прогноз (тренировка)", linestyle="--")
plt.plot(test.index[look_back:], test_predict.flatten(), label="Прогноз (тест)", linestyle="--")
plt.xlabel("Дата")
plt.ylabel("Часы")
plt.title("Прогноз часов работы техники с помощью LSTM")
plt.legend()
plt.grid()
plt.show()
