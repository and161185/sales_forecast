import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import time
import os
from datetime import datetime
import pickle
import json

models_path = "C:\\python_projects\\sales_forecast\\model\\"
encoders_path = "C:\\python_projects\\sales_forecast\\model\\label_encoders.pkl"
directory_normalized = 'C:\\python_projects\\sales_forecast\\data\\normalized'
directory_predictions = 'C:\\python_projects\\sales_forecast\\data\\predictions'

categorical_columns = ['shop', 'goodsCode1c', 'subgroup', 'group', 'category', 'action_type']
xcol = ['price', 'temperature', 'prcp', 'holiday',
        'pre_holiday', 'is_working_day', 'weekend', 'category_avg_price', 'price_level',
        'new',
        'allSalesCount_ma_30', 'allSalesCount_ma_7', 'count_ma_7', 'count_ma_30', 'price_ma_7', 'price_ma_30', 'currencyRate_ma_7',
        'allSalesCount_lag_1', 'allSalesCount_lag_7', 'returns_rate_lag_1', 'sell_ratio_lag_1',
        'sold_out_lag_1', 'average_google_trend_lag_1', 'google_trend_growth_rate_lag_1', 
        'allSalesCount_growth_rate_1', 'allSalesCount_growth_rate_7', 'price_growth_rate_7', 'day_inflation',
        'count_lag_1', 'count_lag_7', 'price_lag_1', 'price_lag_7', 'currencyRate_lag_1', 'count_growth_rate_1', 'count_growth_rate_7', 
        'day', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'day_of_month_sin', 'day_of_month_cos']
ycol_is_sell = ['is_sell']
ycol_count = ['allSalesCount', 'count',  'action_count']

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

def test_on_files(model, test_files, label_encoders):
    """
    Проверяет модель на тестовых данных.

    Аргументы:
    - model: обученная модель.
    - test_files: список файлов для тестирования.

    Выводит метрики модели на тестовых данных.
    """

    ycol = model['ycol']
    model = model['model']
    condition = model['condition']


    # Подготавливаем данные для теста
    result = []
    X_test, y_test = [], []
    predictions = []
    for file_path in test_files:
        df = pd.read_parquet(file_path)

        if condition:
            filtered_df = df.query(condition)
        else:
            filtered_df = df.copy()  # Возвращаем весь DataFrame

        df = prepare_data_for_model(filtered_df, label_encoders)

        X = [df[col].values.astype('int32') for col in categorical_columns]
        X.append(df[xcol].values.astype('float32'))
        y = df[ycol].values.astype('float32')

        X_test.append(X)
        y_test.append(y)

        # Предсказания
        y_pred = model.predict(X)
        predictions.append(y_pred)

        # Добавляем предсказанные значения в DataFrame
        for idx, col in enumerate(ycol):
            df[f"{col}_predicted"] = y_pred[:, idx]

        # Сохраняем DataFrame с предсказаниями
        output_path = os.path.join(directory_predictions, os.path.basename(file_path))
        df.to_parquet(output_path)
        result.append(output_path)

        return result

def load_model_data(model_name):
    # Загружаем модель
    model_path = os.path.join(models_path, model_name + '.keras')
    model = load_model(model_path)
    
    # Загружаем метаданные
    metadata_path = os.path.join(models_path, model_name + '_metadata.json')
    with open(metadata_path, 'r') as f:
        model_data = json.load(f)
    
    model_data['model'] = model
    # Возвращаем модель и метаданные
    return model_data


def load_encoders():
    # Загружаем label_encoders
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)  # Загружаем энкодеры из файла

    return label_encoders

def main():
    os.makedirs(directory_predictions, exist_ok=True)
    
    model_is_sell = load_model_data('is_sell')
    model_count = load_model_data('count')

    label_encoders = load_encoders()

    print("Models and label encoders loaded successfully.")

    # Даты для тестирования
    test_start_date = datetime(2024, 4, 1)
    test_end_date = datetime(2024, 4, 30)
    test_files = get_files_by_date_range(directory_normalized, test_start_date, test_end_date)
    
    temp_files = test_on_files(model_is_sell, test_files, label_encoders)
    test_on_files(model_count, temp_files, label_encoders)

if __name__ == "__main__":
    main()
