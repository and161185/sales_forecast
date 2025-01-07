from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import pandas as pd
import os
from datetime import datetime


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


def prepare_data_for_model(df):
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    embeddings = {}
    for col in categorical_columns:
        unique_values = df[col].nunique()
        embeddings[col] = create_embedding_layer(unique_values, 10)

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, embeddings


def create_embedding_layer(input_dim, output_dim):
    return tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)


def create_model(numerical_columns, categorical_columns, embeddings):
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

    output = layers.Dense(1, activation='linear')(dense2)
    model = Model(inputs=input_layers, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_model_on_files(files_to_load, epochs=1):
    model = None
    embeddings = {}

    for _ in range(epochs):
        for file_path in files_to_load:
            df = pd.read_parquet(file_path)
            df, new_embeddings = prepare_data_for_model(df)

            if model is None:
                embeddings = new_embeddings
                model = create_model(numerical_columns, categorical_columns, embeddings)

            X = [df[col].astype('int32') for col in categorical_columns]
            X.append(df[numerical_columns].astype('float32'))
            y = df[ycol].values

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
