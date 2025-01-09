from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
from datetime import datetime
import time
import pickle


model_path = "C:\\python_projects\\sales_forecast\\model\\model.keras"
encoders_path = "C:\\python_projects\\sales_forecast\\model\\label_encoders.pkl"
directory_test = 'C:\\python_projects\\sales_forecast\\data\\normalized'
directory = "C:\\python_projects\\sales_forecast\\data\\shuffled"

categorical_columns = ['shop', 'goodsCode1c', 'subgroup', 'group', 'category']
xcol = ['price', 'temperature', 'prcp', 'holiday',
        'pre_holiday', 'is_working_day', 'action_avg_price', 'count_ma_7', 'count_ma_30', 'price_ma_7', 'price_ma_30', 'currencyRate_ma_7',
        'count_lag_1', 'count_lag_7', 'price_lag_1', 'price_lag_7', 'currencyRate_lag_1', 'count_growth_rate_1', 'count_growth_rate_7', 
        'day', 'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
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


def create_embedding_layer(input_dim, output_dim):
    print_log("create_embedding_layer")
    return tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)


def create_model(categorical_columns, embeddings):
    print_log("create_model")

    input_layers = []
    embedding_layers = []

    for col in categorical_columns:
        input_layer = layers.Input(shape=(1,), name=col)
        embedding_layer = embeddings[col](input_layer)
        embedding_layers.append(layers.Flatten()(embedding_layer))
        input_layers.append(input_layer)

    numeric_input = layers.Input(shape=(len(xcol),), name='numeric_input')
    input_layers.append(numeric_input)

    numeric_dense = layers.Dense(64, activation='relu')(numeric_input)

    combined = layers.Concatenate()([*embedding_layers, numeric_dense])
    dense1 = layers.Dense(256, activation='relu')(combined)
    dropout1 = layers.Dropout(0.3)(dense1)

    dense2 = layers.Dense(128, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.3)(dense2)

    dense3 = layers.Dense(64, activation='relu')(dropout2)
    dropout3 = layers.Dropout(0.3)(dense3)

    output = layers.Dense(len(ycol), activation='linear')(dropout3)

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
    label_encoders = {}  # Для хранения энкодеров для каждого столбца

    for col, unique_values in unique_values_per_column.items():
        embedding_size = min(50, max(1, int(len(unique_values)**0.5)))
        print_log(f"embedding_size {col} = {embedding_size}")
        input_dim = len(unique_values)
        embeddings[col] = create_embedding_layer(input_dim, embedding_size)  

        # Обучаем LabelEncoder на уникальных значениях
        le = LabelEncoder()
        le.fit(list(unique_values))  # Обучаем энкодер на уникальных значениях
        label_encoders[col] = le  # Сохраняем энкодер для дальнейшего использования

    print_log("got embeddings")
    return embeddings, label_encoders 
            

def train_model_on_files(model, files_to_load, df_val_test, label_encoders, epochs=1):

    print_log("train_model_on_files Learning")
    # Основной цикл обучения
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    df_val_test = prepare_data_for_model(df_val_test, label_encoders)
    X_val = [df_val_test[col].values.astype('int32') for col in categorical_columns]
    X_val.append(df_val_test[xcol].values.astype('float32'))
    y_val = df_val_test[ycol].values.astype('float32')

    for file_path in files_to_load:
        df = pd.read_parquet(file_path)
        df = prepare_data_for_model(df, label_encoders)

        # Формируем входные данные для модели
        X = [df[col].values.astype('int32') for col in categorical_columns]
        X.append(df[xcol].values.astype('float32'))
        y = df[ycol].values.astype('float32')

        # Обучаем модель на текущем файле
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stopping])

    return model


def get_files_by_date_range(directory, start_date, end_date):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_date = datetime.strptime(filename, "%Y-%m.parquet")
            if start_date <= file_date <= end_date:
                files.append(os.path.join(directory, filename))
    return files

def test_model_on_files(model, df, label_encoders):
    """
    Проверяет модель на тестовых данных.

    Аргументы:
    - model: обученная модель.
    - test_files: список файлов для тестирования.

    Выводит метрики модели на тестовых данных.
    """

    # Подготавливаем данные для теста
    X_test, y_test = [], []

    df = prepare_data_for_model(df, label_encoders)

    X = [df[col].values.astype('int32') for col in categorical_columns]
    X.append(df[xcol].values.astype('float32'))
    y = df[ycol].values.astype('float32')

    X_test.append(X)
    y_test.append(y)

    # Оцениваем модель
    print("Evaluating model on test data...")
    for X, y in zip(X_test, y_test):
        loss, mae = model.evaluate(X, y, verbose=1)
        print(f"Test Loss: {loss}, Test MAE: {mae}")


def save_model_and_encoders(model, label_encoders):
    # Сохраняем модель
    model.save(model_path)  # Сохранение модели в файл

    # Сохраняем label_encoders
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)  # Сохраняем энкодеры в файл

def get_all_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def main():
    files_to_load = get_all_files(directory)

    # Даты для тестирования
    test_start_date = datetime(2024, 4, 1)
    test_end_date = datetime(2024, 4, 30)
    test_files = get_files_by_date_range(directory_test, test_start_date, test_end_date)

    test_file = test_files[0]
    df_test = pd.read_parquet(test_file)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    df_val_test = df_test[:int(0.5 * len(files_to_load))]
    df_test = files_to_load[int(0.5* len(files_to_load)):]

    all_files = files_to_load + test_files
    # Создаем LabelEncoders и эмбеддинги

    embeddings, label_encoders = get_embeddings(all_files)

    # Создаем модель
    model = create_model(categorical_columns, embeddings)

    train_model_on_files(model, files_to_load, df_val_test, label_encoders, epochs=3)

    test_model_on_files(model, df_test, label_encoders)

    save_model_and_encoders(model, label_encoders)

if __name__ == "__main__":
    main()
