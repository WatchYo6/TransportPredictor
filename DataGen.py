import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import holidays


def generate_data(filename, period, seasonality, include_holidays):
    start_date = datetime.today() - timedelta(days=period)
    dates = [start_date + timedelta(days=i) for i in range(period)]
    seasonality_factors = {
        1: 0.7,  # Январь
        2: 0.7,  # Февраль
        3: 0.8,  # Март
        4: 1.0,  # Апрель
        5: 1.2,  # Май
        6: 1.3,  # Июнь
        7: 1.3,  # Июль
        8: 1.2,  # Август
        9: 1.1,  # Сентябрь
        10: 1.0, # Октябрь
        11: 0.8, # Ноябрь
        12: 0.7  # Декабрь
    }

    # Список праздничных дней
    russian_holidays = holidays.Russia()
    data = []

    # Генерация данных
    for tr_id in range(1, 201):  # 1-100 самосвалы, 101-200 экскаваторы
        type_factor = 1 if tr_id <= 100 else 1.2  # Экскаваторы работают на 20% больше
        for date in dates:
            # Учет выходных и праздников
            is_weekend_or_holiday = include_holidays and (date.weekday() >= 5 or date in russian_holidays)
            max_hours = 5 if is_weekend_or_holiday else 10

            # Добавляем сезонность
            month = date.month
            seasonal_effect = 1 + seasonality / 100 * (seasonality_factors[month] - 1)
            base_hours = random.uniform(0, max_hours)
            hours = base_hours * seasonal_effect * type_factor

            # Ограничиваем значения
            hours = max(0, min(hours, max_hours))
            data.append({"date": date.strftime("%Y-%m-%d"), "tr_id": tr_id, "hours": round(hours, 1)})

    # Сохранение в файл
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Файл сохранен как {filename}")


# Интерфейс tkinter
def create_gui():
    def on_generate():
        filename = file_name_var.get()
        if not filename.endswith(".csv"):
            filename += ".csv"
        try:
            period = int(period_var.get())
            seasonality = seasonality_var.get()
            include_holidays = holidays_var.get()
            generate_data(filename, period, seasonality, include_holidays)
            messagebox.showinfo("Успех", f"Данные успешно сгенерированы в {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    root = tk.Tk()
    root.title("Генератор данных для ТС")

    # Поля ввода
    tk.Label(root, text="Имя файла").grid(row=0, column=0, padx=10, pady=5, sticky="w")
    file_name_var = tk.StringVar(value="transport_data.csv")
    tk.Entry(root, textvariable=file_name_var, width=30).grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Период (дней)").grid(row=1, column=0, padx=10, pady=5, sticky="w")
    period_var = tk.StringVar(value="30")
    tk.Entry(root, textvariable=period_var, width=30).grid(row=1, column=1, padx=10, pady=5)

    tk.Label(root, text="Сезонность").grid(row=2, column=0, padx=10, pady=5, sticky="w")
    seasonality_var = tk.IntVar(value=50)
    ttk.Scale(root, variable=seasonality_var, from_=0, to=100, orient="horizontal").grid(row=2, column=1, padx=10,
                                                                                         pady=5)

    holidays_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Учитывать выходные и праздники", variable=holidays_var).grid(row=3, column=1, padx=10,
                                                                                            pady=5, sticky="w")

    # Кнопка генерации
    tk.Button(root, text="Сгенерировать", command=on_generate).grid(row=4, column=0, columnspan=2, pady=10)

    root.mainloop()


create_gui()
