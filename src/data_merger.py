import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import gc
import time

last_log_time = time.time()

# Определение типов данных
dtype_leftovers = {
    'shop': 'int8',
    'goodsCode1c': 'str',
    'goodsCode': 'str',
    'subgroup': 'str',
    'group': 'str',
    'category': 'str',
    'price': 'float32',
    'leftovers': 'float32',
    'allSalesCount': 'float32',
    'allSalesAmount': 'float32'
}

dtype_retail_sales = {
    'shop': 'int8',
    'goodsCode': 'str',
    'avg_price': 'float32',
    'count': 'float32',
    'action_type': 'str',
    'amount': 'float32',
    'discount_value': 'float32',
    'action_amount': 'float32',
    'action_count': 'float32',
    'action_avg_price': 'float32'
}

# Пути к файлам
leftovers_path = "C:\\python_projects\\sales_forecast\\data\\leftovers\\"
retail_sales_path = "C:\\python_projects\\sales_forecast\\data\\retail_sales_data\\"
usd_kzt_path = "C:\\python_projects\\sales_forecast\\data\\usd_kzt.xlsx"
weather_path = "C:\\python_projects\\sales_forecast\\data\\weather.xlsx"
holidays_path = "C:\\python_projects\\sales_forecast\\data\\holidays.xlsx"
temp_path = "C:\\python_projects\\sales_forecast\\data\\temp\\"
prepared_path = "C:\\python_projects\\sales_forecast\\data\\prepared\\"

def print_log(message):
    global last_log_time
    
    # Получаем текущее время
    current_time = time.time()
    
    # Рассчитываем время с последнего лога
    time_diff = current_time - last_log_time
    last_log_time = current_time  # Обновляем время последнего лога
    
    print(f"[{int(time_diff)}s] {message}")  # Обновляем строку с временем
    
def remove_zero_data_pairs(df):
    print_log("исправление нулевых цен")

    # 0.0 -> NaN
    df[['price', 'currencyRate']] = df[['price', 'currencyRate']].mask(df[['price', 'currencyRate']] == 0.0)

    # Сортируем и заполняем пропуски
    df.sort_values(['shop', 'goodsCode1c'], kind='mergesort', inplace=True)
    df[['price', 'currencyRate']] = df[['price', 'currencyRate']].ffill().bfill()  # Исправлено!

    # Проверяем "нулевые" группы
    zero_mask = df.groupby(['shop', 'goodsCode1c'])['price'].transform('sum') == 0

    # Логи удаленных групп
    removed = df.loc[zero_mask, ['shop', 'goodsCode1c']].drop_duplicates()
    if not removed.empty:
        print_log("удалены группы: " + ", ".join(
            [f"shop={row['shop']}, goodsCode1c={row['goodsCode1c']}" for _, row in removed.iterrows()]
        ))

    # Возвращаем отфильтрованные данные
    return df[~zero_mask]



def load_currency_data(currency_path):
    """Загружает данные о курсах валют и преобразует их в словарь."""
    data = pd.read_excel(currency_path)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='%d.%m.%Y')
    return data.set_index('Date')['USD'].to_dict()

def load_weather_data(path):
    """Загружает данные о температуре и преобразует их в словарь."""
    data = pd.read_excel(path)
    data['date'] = pd.to_datetime(data['date'], errors='coerce', format='%Y-%m-%d')
    return data.set_index('date')[['tavg', 'prcp']].to_dict(orient='index')

def load_holidays_data(path):
    """Загружает данные о праздниках и преобразует их в словарь."""
    data = pd.read_excel(path)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', format='%d.%m.%Y')
    return data.set_index('Date')[['Holiday', 'Pre_holiday', 'Rescheduled_working_day']].to_dict(orient='index')


def read_file_by_date_range(path, start_date, end_date, dtype, read_prev_month, sort_values = False, ext = 'csv'):
    print_log(f"чтение {path} на дату {start_date}")
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Получаем путь к файлам для текущего и предыдущего месяца
    current_month_file = os.path.join(path, f"{start_date.strftime('%Y-%m')}.{ext}")

    previous_month_file = os.path.join(path, f"{(start_date - pd.DateOffset(months=1)).strftime('%Y-%m')}.{ext}")    

    previous_month_data = pd.DataFrame()  # Если файла нет, создаем пустой DataFrame
    if read_prev_month:
        try:
            if ext=='csv':
                previous_month_data = pd.read_csv(previous_month_file, parse_dates=['operDay'], dtype = dtype, engine='pyarrow')            
            if ext=='parquet':
                previous_month_data = pd.read_parquet(previous_month_file)
        except:
            print_log("файл " + previous_month_file + " не найден")      

    # Читаем файл
    if ext=='csv':
        current_month_data = pd.read_csv(current_month_file, parse_dates=['operDay'], dtype = dtype, engine='pyarrow')
    if ext=='parquet':
        current_month_data = pd.read_parquet(current_month_file)

    # Объединяем данные за текущий и предыдущий месяц
    combined_data = pd.concat([previous_month_data, current_month_data], ignore_index=True)

    if sort_values:
        print_log("сортировка")
        combined_data.sort_values(
            by=['shop', 'goodsCode', 'operDay'], 
            ascending=[True, True, True], 
            inplace=True, 
            kind='quicksort'
        )
    

    return combined_data

def add_sales(data, retail_sales):
    print_log("Добавление данных продаж")
    retail_sales['operDay'] = pd.to_datetime(retail_sales['operDay'])

    merged_data = pd.merge(
        data,
        retail_sales,
        on=['operDay', 'shop', 'goodsCode1c'],
        how='left'
    )

    merged_data[['avg_price', 'count']] = merged_data[['avg_price', 'count']].fillna(0)
    merged_data.loc[merged_data['avg_price'] == 0, 'avg_price'] = merged_data['price']

    return merged_data


def add_currency_rate(data, dict):
    print_log("Добавление курсов валют")
    data['currencyRate'] = data['operDay'].map(dict)
    return data

def add_weather(data, dict):
    """Обновляет данные о температуре и осадках в основном DataFrame."""
    print_log("Добавление данных о погоде")

    # Преобразуем dict в DataFrame
    weather_df = pd.DataFrame.from_dict(dict, orient='index')
    weather_df.index.name = 'date'

    # Объединяем weather_data с основным DataFrame по дате
    data = data.merge(weather_df, left_on='operDay', right_index=True, how='left')

    # Заполняем NaN значениями
    data[['tavg', 'prcp']] = data[['tavg', 'prcp']].fillna(0)

    # Переименовываем столбцы
    data = data.rename(columns={
        'tavg': 'temperature',
        'prcp': 'prcp'
    })

    return data
    
def add_holidays(data, holidays_dict):
    print_log("Добавление производственного календаря")
    
    # Преобразуем holidays_dict в DataFrame
    holidays_df = pd.DataFrame.from_dict(holidays_dict, orient='index')
    holidays_df.index.name = 'Date'
    
    # Объединяем holiday_data с основным DataFrame по дате
    data = data.merge(holidays_df, left_on='operDay', right_index=True, how='left')
    
    # Заполняем NaN значениями
    data[['Holiday', 'Pre_holiday', 'Rescheduled_working_day']] = data[['Holiday', 'Pre_holiday', 'Rescheduled_working_day']].fillna(0)
    
    # Переименовываем столбцы
    data = data.rename(columns={
        'Holiday': 'holiday',
        'Pre_holiday': 'pre_holiday',
        'Rescheduled_working_day': 'rescheduled_working_day'
    })

    print_log("Определение рабочих дней")
    # Векторизированное вычисление рабочего дня
    data['is_working_day'] = (
        (data['holiday'] == 0) &
        (
            (data['day_of_week'].isin([6, 7]) & (data['rescheduled_working_day'] == 1)) |
            (~data['day_of_week'].isin([6, 7]))
        )
    ).astype(int)

    # Удаляем столбец
    data = data.drop(columns=['rescheduled_working_day'])

    return data

def fast_shift_and_round(group, shifts, decimals):
    arr = group.to_numpy()
    result = []
    for s in shifts:
        shifted = np.roll(arr, s)
        if s > 0:
            shifted[:s] = np.nan  # Заполняем начало NaN
        result.append(np.round(shifted, decimals))
    return np.column_stack(result)

def add_trends_dynamics(data):
    print_log("Подготовка к расчету трендов")

    # Проверяем, нужно ли сортировать
    if not data[['shop', 'goodsCode', 'operDay']].sort_values(['shop', 'goodsCode', 'operDay']).equals(data):
        data = data.sort_values(by=['shop', 'goodsCode', 'operDay'])

    grouped = data.groupby(['shop', 'goodsCode'], group_keys=False, observed=True)

    # Считаем скользящие средние и лаги за один проход
    print_log("Считаем скользящие средние")
    #скользящее среднее по количеству не можем считать от текущей строки, т.к. на сегодня продажи неизвестны, делаем shift(1)
    data['count_ma_7'] = grouped['count'].shift(1).rolling(7, min_periods=1).mean().round(3).values
    data['count_ma_30'] = grouped['count'].shift(1).rolling(30, min_periods=1).mean().round(3).values
    data['allSalesCount_ma_7'] = grouped['allSalesCount'].shift(1).rolling(7, min_periods=1).mean().round(3).values
    data['allSalesCount_ma_30'] = grouped['allSalesCount'].shift(1).rolling(30, min_periods=1).mean().round(3).values
    
    data['price_ma_7'] = grouped['price'].rolling(7, min_periods=1).mean().round(3).values
    data['price_ma_30'] = grouped['price'].rolling(30, min_periods=1).mean().round(3).values
    data['currencyRate_ma_7'] = grouped['currencyRate'].rolling(7, min_periods=1).mean().round(3).values

    # Считаем лаги
    print_log("Считаем лаги")
    data['count_lag_1'] = grouped['count'].shift(1).round(3).values
    data['count_lag_7'] = grouped['count'].shift(7).round(3).values
    data['allSalesCount_lag_1'] = grouped['allSalesCount'].shift(1).round(3).values
    data['allSalesCount_lag_7'] = grouped['allSalesCount'].shift(7).round(3).values
    data['price_lag_1'] = grouped['price'].shift(1).round(3).values
    data['price_lag_7'] = grouped['price'].shift(7).round(3).values
    data['currencyRate_lag_1'] = grouped['currencyRate'].shift(1).round(3).values

    # Вычисляем темпы роста
    lags = [1, 7]
    print_log("Вычисляем темпы роста")
    for col in ['allSalesCount', 'count']:
        for lag in lags:
            lag_col = f'{col}_lag_{lag}'
            growth_rate_col = f"{col}_growth_rate_{lag}"

            diff = data[lag_col] - grouped[col].shift(lag*2).round(3).values
            data[growth_rate_col] = ((diff / data[lag_col]).fillna(0).replace([np.inf, -np.inf], 0) * 100).round(3)

    return data

def validate_columns(df, columns):
    print_log("проверка колонок")
    
    # Проверяем для всех указанных колонок: NaN или 0
    mask_invalid = df[columns].isna() | (df[columns] == 0)
    
    # Находим строки, где хотя бы в одной колонке есть некорректные значения
    invalid_rows = df[mask_invalid.any(axis=1)]
    
    # Если есть некорректные строки, выводим их
    if not invalid_rows.empty:
        for idx, row in invalid_rows.iterrows():
            print_log(f"Ошибка в строке {idx + 2}: {row.to_dict()}")
        raise ValueError("Проверка не пройдена: найдены некорректные значения.")

def main():
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    #end_date = "2024-01-31"
    

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    currency_dict = load_currency_data(usd_kzt_path)
    weather_dict = load_weather_data(weather_path)
    holidays_dict = load_holidays_data(holidays_path)

    os.makedirs(temp_path, exist_ok=True)

    # Итерация по месяцам
    current_date = start_date_dt
    while current_date <= end_date_dt:
        # Определяем начало и конец текущего месяца
        month_start = current_date.strftime("%Y-%m-%d")
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Переход к следующему месяцу
        month_end = (next_month - timedelta(days=next_month.day)).strftime("%Y-%m-%d")

        print_log(f"обработка данных за {month_start} - {month_end}")

        # Читаем данные за текущий месяц
        data = read_file_by_date_range(leftovers_path, month_start, month_end, dtype_leftovers, False, True)
        data['operDay'] = pd.to_datetime(data['operDay'])

        print_log("Добавление дополнительных столбцов")
        # Добавляем year, month, day_of_month, day_of_week
        data['year'] = data['operDay'].dt.year
        data['month'] = data['operDay'].dt.month
        data['day_of_month'] = data['operDay'].dt.day
        data['day_of_week'] = data['operDay'].dt.weekday + 1

        data = add_currency_rate(data, currency_dict)
        data = add_weather(data, weather_dict)
        data = add_holidays(data, holidays_dict)    

        data = remove_zero_data_pairs(data) 

        retail_sales = read_file_by_date_range(retail_sales_path, month_start, month_end, dtype_retail_sales, False, False, 'parquet')
        data = add_sales(data, retail_sales)
        
        data['leftovers'] = data[['leftovers', 'allSalesCount']].max(axis=1)

        columns_for_validation = ["price", "currencyRate"]
        validate_columns(data, columns_for_validation)

        # Сохраняем DataFrame в файл
        file_name = f"{month_start[:7]}.parquet"  # Берём только год-месяц из строки даты
        file_path = os.path.join(temp_path, file_name)        
        print_log(f"сохраняем файл: {file_path}") 
        data.to_parquet(file_path, index=False)
        print_log(f"Файл сохранён: {file_path}")   

        # Переходим к следующему месяцу
        current_date = next_month.replace(day=1)

        del retail_sales, data
        gc.collect()


    current_date = start_date_dt
    while current_date <= end_date_dt:

        # Определяем начало и конец текущего месяца
        month_start = current_date.strftime("%Y-%m-%d")
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Переход к следующему месяцу
        month_end = (next_month - timedelta(days=next_month.day)).strftime("%Y-%m-%d")

        print_log(f"расчет периодических показателей за {month_start} - {month_end}")
        data = read_file_by_date_range(temp_path, month_start, month_end, None, True, False, "parquet")

        data = add_trends_dynamics(data)

        file_name = f"{month_start[:7]}.parquet"  # Берём только год-месяц из строки даты
        file_path = os.path.join(prepared_path, file_name)

        # Сохраняем DataFrame в файл
        data = data[data['operDay'] >= month_start]
        #для понимания максимального спроса на нулевые остатки не смотрим, в обучении вообще не используем
        data = data[data['leftovers'] > 0]

        columns_to_fill_zero = [
            'count', 'amount', 'discount_value', 'action_amount', 'action_count', 
            'action_avg_price', 'count_ma_7', 'count_ma_30', 
            'count_lag_1', 'count_lag_7', 'price_lag_1', 'price_lag_7',
            'currencyRate_ma_7', 'currencyRate_lag_1',
            'allSalesCount_ma_7', 'allSalesCount_ma_30', 'allSalesCount_lag_1', 'allSalesCount_lag_7'
        ]

        # Список столбцов для заполнения пустыми строками
        columns_to_fill_empty = ['action_type']

        # Заполнение NaN в указанных столбцах
        data[columns_to_fill_zero] = data[columns_to_fill_zero].fillna(0)
        data[columns_to_fill_empty] = data[columns_to_fill_empty].fillna("")

        # Поиск столбцов с NaN
        columns_with_nan = data.columns[data.isna().any()]

        # Если есть столбцы с NaN, выбрасываем ошибку
        if not columns_with_nan.empty:
            error_message = "Найдены столбцы с NaN: " + ", ".join(columns_with_nan)
            raise ValueError(error_message)
        else:
            print_log("NaN отсутствуют.")

        data.to_parquet(file_path, index=False)  # index=False, чтобы не сохранять индексы
        print_log(f"Файл сохранён: {file_path}")

        # Переходим к следующему месяцу
        current_date = next_month.replace(day=1)

        del data
        gc.collect()

if __name__ == "__main__":
    main()

