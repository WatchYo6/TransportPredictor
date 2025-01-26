# TransportPredictor
Программа для прогнозирования износа (пробега) транспорта
## Генерация датасета
Так как реальных данных у меня сейчас нет, нужно "придумать" их самому.
### Структура данных
Допустим, у нас есть таблица transport в ней есть данные о транспорте:\
id - уникальный номер ТС;\
type - тип ТС (экскаватор, самосвал и тд);\
maintenance - норматив по обслуживанию (через какой пробег требуется обслуживание);\
и т.д. - данные из этой таблицы не очень важны, требуется только знать какому типу (type) какое ТС (id) принадлежит.\
В таблице есть 100 самосвалов (type - 1) и 100 экскаваторов (type - 2).\
Они упорядочены - сначала самосвалы (id 1-100), потом экскаваторы. В реальный данных типов транспорта (а скорее моделей) будет больше, но
для простого тестирования моделей прогнозирования это не очень важно.\
_Также нужно будет учесть, что у разных категорий ТС могут быть разные способы измерения работы:_ \
_Пробег в км_ \
_Моточасы_ \
_Если график ремонта составляется по м/ч, то нужно переводить км пробега в м/ч (например, через средний расход топлива)
если же для разных ТС используются разные величины, то лучше обучить модели отдельно._
### Создание данных
Датасет CSV формата со структурой:\
date: дата записи в формате yyyy-mm-dd;\
tr_id - id ТС для которого эта запись сделана;\
hours - сколько часов отработал за этот день;\
Максимальный рабочий день - 10 часов, то есть техника может работать от 0 часов (простой) до 10 часов каждый день (эти величины можно поменять).\
Также нужно учесть выходные и официальные праздники в РФ, предполагается что в это время техника меньше работает.\
**Сезонность** - возможно наличие фактора сезонности, так как дорожные и строительные работы преимущественно проводятся в теплое время года (а возможно и нет - на реальных данных это будет видно).\
Для генерации используется файл DataGen.py, там можно отрегулировать сезонность, а также учесть выходные и праздники.\
**Итог генерации:** файл transport_data.csv
![изображение](https://github.com/user-attachments/assets/d2c5d7ae-5bc4-42fb-ad69-0c0c6e902ac9)
![изображение](https://github.com/user-attachments/assets/2b6bada4-567d-4432-816c-6fc8c536c3fa)
На этих графиках можно наблюдать сезонность данных.
## Выбор модели прогнозирования
Полученный временной ряд **нестационарен**, так как есть сезонные колебания, а значит модель для прогноза должна уметь работать с нелинейными данными. \
Возможные варианты: 
1. SARIMAX
2. LSTM
3. XGBoost
4. Prophet
### Тест моделей
#### LSTM
![изображение](https://github.com/user-attachments/assets/5f3be821-7a6f-4683-ab17-c4e937b42bab)
MAE: 68.37\
RMSE: 120.83\
MAPE: 9.14%\
Время обучения модели: 14.37 секунд\
Время предсказания модели: 0.63 секунд\
#### XGBoost
![изображение](https://github.com/user-attachments/assets/18c6d873-4d97-4ad4-a9b9-3f14a891628c)
MAE (XGBoost): 56.59\
RMSE (XGBoost): 103.49\
MAPE (XGBoost): 7.49%\
Время обучения модели: 0.26 секунд\
Время предсказания модели: 0.00 секунд\
#### Prophet
![изображение](https://github.com/user-attachments/assets/93a5835d-5617-4c6c-a4cf-85d387408121)
MAE: 62.71\
RMSE: 106.14\
MAPE: 9.16%\
Время обучения модели: 0.64 секунд\
Время предсказания модели: 0.36 секунд\
#### SARIMAX
У SARIMAX слишком долгое время обучения (может занять около часа), возможно проведу с ней тест позже.
## Итог
XGBoost стал лучшим и по точности предсказания, и по скорости.\
Также для расширеня теста можно будет протестировать:
- Catboost
- SARIMAX

