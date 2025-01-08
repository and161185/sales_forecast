import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import time
import os
from datetime import datetime
import pickle

model_path = "C:\\python_projects\\sales_forecast\\model\\model.keras"
encoders_path = "C:\\python_projects\\sales_forecast\\model\\label_encoders.pkl"
directory_normalized = 'C:\\python_projects\\sales_forecast\\data\\normalized'

categorical_columns = ['shop', 'goodsCode1c', 'subgroup', 'group', 'category']
numerical_columns = ['price', 'allSalesCount', 'temperature', 'prcp', 'holiday',
                     'pre_holiday', 'is_working_day', 'avg_price', 'count', 'amount', 'discount_value', 'action_amount', 
                     'action_count', 'action_avg_price', 'count_ma_7', 'count_ma_30', 'price_ma_7', 'price_ma_30', 'currencyRate_ma_7',
                     'count_lag_1', 'count_lag_7', 'price_lag_1', 'price_lag_7', 'currencyRate_lag_1', 'count_growth_rate_1', 'count_growth_rate_7', 
                     'price_growth_rate_1', 'price_growth_rate_7', 'day', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos', 
                     'day_of_month_sin', 'day_of_month_cos']
xcol = ['price', 'temperature', 'prcp', 'holiday',
                     'pre_holiday', 'is_working_day', 'action_avg_price', 'count_ma_7', 'count_ma_30', 'price_ma_7', 'price_ma_30', 'currencyRate_ma_7',
                     'count_lag_1', 'count_lag_7', 'price_lag_1', 'price_lag_7', 'currencyRate_lag_1', 'count_growth_rate_1', 'count_growth_rate_7', 
                     'price_growth_rate_1', 'price_growth_rate_7', 'day', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
                     'day_of_month_sin', 'day_of_month_cos']
ycol = ['allSalesCount', 'count',  'action_count']

last_log_time = time.time()

def print_log(message):
    global last_log_time
    
    # Получаем текущее время
    current_time = time.time()
    
    # Рассчитываем время с последнего лога
    time_diff = current_time - last_log_time
    last_log_time = current_time  # Обновляем время последнего лога
    
    print(f"[{int(time_diff)}s] {message}")  # Обновляем строку с временем


def get_files_by_date_range(directory, start_date, end_date):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_date = datetime.strptime(filename, "%Y-%m.parquet")
            if start_date <= file_date <= end_date:
                files.append(os.path.join(directory, filename))
    return files

def prepare_data_for_model(df, label_encoders):
    """
    Подготавливает данные для модели, обрабатывая категориальные и числовые данные.

    Аргументы:
    - df: DataFrame с данными.
    - label_encoders: словарь с объектами LabelEncoder для категориальных колонок.

    Возвращает:
    - DataFrame с обработанными данными.
    """

    print_log("prepare_data_for_model")

    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])

    print_log("prepared data_for_model")
    return df

def test_model_on_files(model, test_files, label_encoders):
    """
    Проверяет модель на тестовых данных.

    Аргументы:
    - model: обученная модель.
    - test_files: список файлов для тестирования.

    Выводит метрики модели на тестовых данных.
    """

    # Подготавливаем данные для теста
    X_test, y_test = [], []
    for file_path in test_files:
        df = pd.read_parquet(file_path)
        df = prepare_data_for_model(df, label_encoders)

        X = [df[col].values.astype('int32') for col in categorical_columns]
        X.append(df[numerical_columns].values.astype('float32'))
        y = df[ycol].values.astype('float32')

        X_test.append(X)
        y_test.append(y)

    # Оцениваем модель
    print("Evaluating model on test data...")
    for X, y in zip(X_test, y_test):
        loss, mae = model.evaluate(X, y, verbose=1)
        print(f"Test Loss: {loss}, Test MAE: {mae}")

def load_model_and_encoders():
    # Загружаем модель
    model = load_model(model_path)

    # Загружаем label_encoders
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)  # Загружаем энкодеры из файла

    print("Model and label encoders loaded successfully.")
    return model, label_encoders

def main():
    model, label_encoders = load_model_and_encoders()

    # Даты для тестирования
    test_start_date = datetime(2024, 4, 1)
    test_end_date = datetime(2024, 4, 30)
    test_files = get_files_by_date_range(directory_normalized, test_start_date, test_end_date)
    
    test_model_on_files(model, test_files, label_encoders)

if __name__ == "__main__":
    main()
