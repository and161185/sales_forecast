from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import os
from datetime import datetime
import time


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

def create_label_encoders(files_to_load, categorical_columns):
    print_log("create_label_encoders")
    label_encoders = {}
    unique_values_per_column = {col: set() for col in categorical_columns}

    # Сбор уникальных значений из всех файлов
    for file_path in files_to_load:
        print_log(f"create_label_encoders {file_path}")
        df = pd.read_parquet(file_path)
        for col in categorical_columns:
            if col in df.columns:
                unique_values_per_column[col].update(df[col].dropna().unique())

    # Создание LabelEncoder на основе всех уникальных значений
    for col, unique_values in unique_values_per_column.items():
        print_log(f"create_label_encoders LabelEncoder {col}")
        le = LabelEncoder()
        le.fit(list(unique_values))
        label_encoders[col] = le

    print_log("created label_encoders")

    return label_encoders

def prepare_data_for_model(df, label_encoders, scaler, fit=False):
    """
    Подготавливает данные для модели, обрабатывая категориальные и числовые данные.

    Аргументы:
    - df: DataFrame с данными.
    - label_encoders: словарь с объектами LabelEncoder для категориальных колонок.
    - scaler: объект StandardScaler для числовых данных.
    - fit: если True, обучает Scaler на данных (первый проход).

    Возвращает:
    - DataFrame с обработанными данными.
    """

    print_log("prepare_data_for_model")

    for col in categorical_columns:
        df[col] = label_encoders[col].transform(df[col])

    if fit:
        scaler.partial_fit(df[numerical_columns])
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    print_log("prepared data_for_model")
    return df


def create_embedding_layer(input_dim, output_dim):
    print_log("create_embedding_layer")
    return tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)


def create_model(numerical_columns, categorical_columns, embeddings):
    print_log("create_model")

    input_layers = []
    embedding_layers = []

    for col in categorical_columns:
        input_layer = layers.Input(shape=(1,), name=col)
        embedding_layer = embeddings[col](input_layer)
        embedding_layers.append(layers.Flatten()(embedding_layer))
        input_layers.append(input_layer)

    numeric_input = layers.Input(shape=(len(numerical_columns),), name='numeric_input')
    input_layers.append(numeric_input)

    numeric_dense = layers.Dense(64, activation='relu')(numeric_input)

    combined = layers.Concatenate()([*embedding_layers, numeric_dense])
    dense1 = layers.Dense(128, activation='relu')(combined)
    dense2 = layers.Dense(64, activation='relu')(dense1)

    output = layers.Dense(len(ycol), activation='linear')(dense2)

    model = Model(inputs=input_layers, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print_log("created model")

    return model

def get_embeddings(files_to_load):
    """
    Аргументы:
    - files_to_load: список путей к файлам.
    """
    # Хранение уникальных значений для каждой колонки

    print_log("get_embeddings")

    unique_values_per_column = {col: set() for col in categorical_columns}

    # Перебор файлов и накопление уникальных значений
    for file_path in files_to_load:
        print_log(f"get_embeddings {file_path}")
        df = pd.read_parquet(file_path)
        for col in categorical_columns:
            if col in df.columns:
                unique_values_per_column[col].update(df[col].dropna().unique())
    
    # Создание эмбеддингов
    embeddings = {}
    for col, unique_values in unique_values_per_column.items():
        embedding_size = min(50, max(1, int(len(unique_values)**0.5)))
        input_dim = len(unique_values)
        embeddings[col] = create_embedding_layer(input_dim, embedding_size)  

    print_log("got embeddings")
    return embeddings  
            

def train_model_on_files(files_to_load, epochs=1):

    # Создаем LabelEncoders и эмбеддинги
    label_encoders = create_label_encoders(files_to_load, categorical_columns)
    embeddings = get_embeddings(files_to_load)

    # Создаем модель
    model = create_model(numerical_columns, categorical_columns, embeddings)

    # Создаем StandardScaler для масштабирования числовых данных
    scaler = StandardScaler()

    print_log("train_model_on_files StandardScaler")

    # Первый проход для обучения StandardScaler
    for file_path in files_to_load:
        df = pd.read_parquet(file_path)
        df = prepare_data_for_model(df, label_encoders, scaler, fit=True)

    print_log("train_model_on_files Learning")
    # Основной цикл обучения
    for _ in range(epochs):
        for file_path in files_to_load:
            df = pd.read_parquet(file_path)
            df = prepare_data_for_model(df, label_encoders, scaler, fit=False)

            # Формируем входные данные для модели
            X = [df[col].values.astype('int32') for col in categorical_columns]
            X.append(df[numerical_columns].values.astype('float32'))
            y = df[ycol].values.astype('float32')

            # Обучаем модель на текущем файле
            model.fit(X, y, epochs=1, batch_size=64, verbose=1)

    return model


def get_files_by_date_range(directory, start_date, end_date):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_date = datetime.strptime(filename, "%Y-%m.parquet")
            if start_date <= file_date <= end_date:
                files.append(os.path.join(directory, filename))
    return files


def main():
    directory = 'C:\\python_projects\\sales_forecast\\data\\normalized'
    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 7, 31)
    files_to_load = get_files_by_date_range(directory, start_date, end_date)
    train_model_on_files(files_to_load)


if __name__ == "__main__":
    main()
