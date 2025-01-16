import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import gc
import time

range_file = "C:\\python_projects\\sales_forecast\\data\\global_ranges.parquet"
prepared_path = "C:\\python_projects\\sales_forecast\\data\\prepared\\"
normalized_path = "C:\\python_projects\\sales_forecast\\data\\normalized\\"

last_log_time = time.time()

def print_log(message):
    global last_log_time
    
    # Получаем текущее время
    current_time = time.time()
    
    # Рассчитываем время с последнего лога
    time_diff = current_time - last_log_time
    last_log_time = current_time  # Обновляем время последнего лога
    
    print(f"[{int(time_diff)}s] {message}")  # Обновляем строку с временем

def read_prepared(current_month_file):
    month_data = pd.read_parquet(current_month_file)
    month_data["action_type"] = month_data["action_type"].replace([0, "0"], "")
    return month_data

# Логарифмическая трансформация с учетом сдвига
def safe_log_transform_with_shift(series, shift=None):
    if shift is None:
        shift = abs(series.min()) + 1 if series.min() < 0 else 0
    return np.log1p(series + shift), shift

# Преобразование циклических признаков
# Например, для дней недели, месяцев, дней в месяце
# Это позволит модели учитывать цикличность этих данных
# Угол передается в виде отношения к полному кругу (например, 2*pi * значение / максимум)
# Преобразование циклических признаков с учетом реального количества дней в месяце
def add_cyclic_features_with_days(df, column, month_column):
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }
    df["days_in_month"] = df[month_column].map(days_in_month)
    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / df["days_in_month"])
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / df["days_in_month"])
    df.drop(columns=["days_in_month"], inplace=True)
    return df

# Преобразование циклических признаков для фиксированных диапазонов
def add_cyclic_features(df, column, min_value, max_value):
    print_log(f"добавление циклических признаков {column}")
    # Нормализуем столбец в диапазон [0, 1] с использованием min_value и max_value
    normalized_column = (df[column] - min_value) / (max_value - min_value)
    # Применяем синус и косинус
    df[f"{column}_sin"] = np.sin(2 * np.pi * normalized_column)
    df[f"{column}_cos"] = np.cos(2 * np.pi * normalized_column)
    return df

def add_normalized_day_column(df, start_year=2010, end_year=2100):
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)
    total_days = (end_date - start_date).days
    df["day"] = (pd.to_datetime(df["operDay"]) - start_date).dt.days / total_days
    return df

# Проверка и загрузка диапазонов и сдвигов
def load_ranges(file):
    if os.path.exists(file):
        # Загружаем диапазоны из Parquet
        ranges_df = pd.read_parquet(file)

        # Убедимся, что индексы в порядке
        ranges_df.reset_index(drop=True, inplace=True)

        return ranges_df

    # Если файл отсутствует, возвращаем пустой DataFrame
    return pd.DataFrame()

# Сохранение диапазонов и сдвигов в файл
def save_ranges(file, ranges_df):
    try:
        ranges_df.to_parquet(file, index=False)
    except IOError as e:
        raise RuntimeError(f"Failed to save ranges to file: {e}")

def normalize_data(df, numeric_cols, group_cols, normalize_ranges, new_ranges):
    print_log("начало нормализации")

    # Считаем min/max по группам
    agg = df.groupby(group_cols)[numeric_cols].agg(["min", "max"]).reset_index()

    # Преобразуем в плоскую структуру
    agg.columns = group_cols + [f"{col}_{stat}" for col in numeric_cols for stat in ["min", "max"]]

    print_log("min/max рассчитан")

    # Мержим результат агрегации с исходным df
    merged = df.merge(agg, on=group_cols, how="left")

    # Мержим с глобальными диапазонами для нормализации
    merged = merged.merge(normalize_ranges, on=group_cols, how="left", suffixes=("", "_global"))

    print_log("данные объединены")

    # Нормализуем данные
    for col in numeric_cols:
        cmin_global = f"{col}_min_global"
        cmax_global = f"{col}_max_global"

        # Нормализуем по глобальным диапазонам (normalize_ranges)
        merged[col] = (merged[col] - merged[cmin_global]) / (merged[cmax_global] - merged[cmin_global] + 1e-8)

        gc.collect()
    
    print_log("нормализация выполнена")

    # Обновляем ranges
    # Собираем min/max по группам сразу для всех числовых колонок
    group_mins = merged.groupby(group_cols)[[f"{col}_min" for col in numeric_cols]].min()
    group_maxs = merged.groupby(group_cols)[[f"{col}_max" for col in numeric_cols]].max()
    print_log("рассчитаны мин/макс по группам")

    # Объединяем в один DataFrame
    group_stats = group_mins.join(group_maxs, lsuffix="_min", rsuffix="_max")
    print_log("мин/макс объединены")

    # Если ranges пустые — инициализируем их
    if new_ranges.empty:
        new_ranges = group_stats.copy()
    else:
        # Добавляем новые минимумы и максимумы
        for col in numeric_cols:
            new_ranges[f"{col}_min"] = np.minimum(
                new_ranges.get(f"{col}_min", np.inf), group_stats[f"{col}_min"]
            )
            new_ranges[f"{col}_max"] = np.maximum(
                new_ranges.get(f"{col}_max", -np.inf), group_stats[f"{col}_max"]
            )
   
    # Объединяем все колонки для удаления в один список
    columns_to_drop = list(
        merged.filter(regex="(_min|_max|_min_global|_max_global|_global)$").columns
    ) + ["day_of_week", "day_of_month", "year", "goodsCode", "currencyRate", "leftovers"]  # Добавляем фиксированные колонки

    # Удаляем все ненужные колонки за один вызов
    merged.drop(columns=columns_to_drop, inplace=True)

    print_log("нормализация завершена")

    return merged

def calculate_dicts(prepared_path, numeric_columns, group_cols, existing_ranges=None):

    # Загружаем существующие ranges, если они есть
    if existing_ranges is not None:
        ranges_df = pd.DataFrame.from_dict(existing_ranges, orient="index")
        ranges_df.index = pd.MultiIndex.from_tuples(ranges_df.index, names=group_cols)
    else:
        ranges_df = pd.DataFrame()

    # Перебираем файлы
    for file_name in os.listdir(prepared_path):
        if file_name.endswith(".parquet"):
            file_path = os.path.join(prepared_path, file_name)
            print_log(f"Обработка файла: {file_path}")

            try:
                # Читаем файл
                df = pd.read_parquet(file_path)

                # Агрегируем данные
                agg = df.groupby(group_cols)[numeric_columns].agg(["min", "max"])
                agg.columns = ["_".join(col) for col in agg.columns]  # Преобразуем мультииндекс в плоские имена колонок

                # Если диапазоны уже есть, объединяем их
                if not ranges_df.empty:
                    # Выполняем join по индексам
                    merged = ranges_df.join(agg, how="outer", rsuffix="_new")

                    # Пересчитываем минимумы и максимумы
                    for col in numeric_columns:
                        # Считаем минимум и максимум отдельно для каждого параметра
                        merged[f"{col}_min"] = merged[[f"{col}_min", f"{col}_min_new"]].min(axis=1, skipna=True)
                        merged[f"{col}_max"] = merged[[f"{col}_max", f"{col}_max_new"]].max(axis=1, skipna=True)

                    # Удаляем временные колонки
                    merged.drop(columns=[f"{col}_min_new" for col in numeric_columns] +
                                         [f"{col}_max_new" for col in numeric_columns],
                                inplace=True)
                else:
                    # Если данных нет, просто присваиваем текущий результат
                    merged = agg

                ranges_df = merged

            except Exception as e:
                print_log(f"Ошибка обработки файла {file_path}: {e}")
                continue

    # Сбрасываем индекс, чтобы добавить поля группировки в итоговый DataFrame
    ranges_df = ranges_df.reset_index()

    # Добавляем дополнительные колонки для группы
    ranges_df["shop"] = ranges_df["shop"].astype("int8")  # или если не строка
    ranges_df["goodsCode1c"] = ranges_df["goodsCode1c"].astype("string")  # или если не строка

    # Возвращаем DataFrame
    return ranges_df

def check_ranges_for_retraining(old_ranges, new_ranges, numeric_cols, threshold=0.1):
    # Сравниваем только числовые диапазоны из numeric_cols
    for col in numeric_cols:
        old_min = old_ranges[f"{col}_min"].values
        old_max = old_ranges[f"{col}_max"].values
        new_min = new_ranges[f"{col}_min"].values
        new_max = new_ranges[f"{col}_max"].values

        # Проверяем, изменились ли диапазоны слишком сильно
        min_diff = np.abs(old_min - new_min) / (np.abs(old_min) + 1e-8)
        max_diff = np.abs(old_max - new_max) / (np.abs(old_max) + 1e-8)

        if np.any(min_diff > threshold) or np.any(max_diff > threshold):
            print_log(f"Предупреждение: диапазоны для {col} изменились слишком сильно. Возможно, требуется переобучение сети.")

# main функция
def main():

    numeric_columns = [
        "price", "allSalesCount", "allSalesAmount", "currencyRate", "temperature",
        "prcp", "avg_price", "count", "amount", "discount_value",
        "action_amount", "action_count", "action_avg_price", "count_ma_7", "count_ma_30",
        "price_ma_7", "price_ma_30", "currencyRate_ma_7", "count_lag_1", "count_lag_7", "price_lag_1", "price_lag_7",
        "currencyRate_lag_1", "count_growth_rate_1", "count_growth_rate_7",
        "allSalesCount_ma_7", "allSalesCount_ma_30", "allSalesCount_lag_1", "allSalesCount_lag_7",
        "category_avg_price", "price_growth_rate_7", "day_inflation",
        "price_level", "sell_ratio", "average_google_trend",
        "average_google_trend_lag_1"
    ]
    
    group_cols = ["shop", "goodsCode1c"]

    start_date = "2024-01-01"
    end_date = "2024-12-31"

    os.makedirs(normalized_path, exist_ok=True)

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Читаем существующие диапазоны, если файл есть
    existing_ranges = load_ranges(range_file)
    # Вычисляем глобальные диапазоны

    # Если существуют диапазоны, используем их как глобальные, иначе вычисляем
    if not existing_ranges.empty:
        global_ranges = existing_ranges
        print_log("Загружены существующие диапазоны.")
    else:
        print_log("Вычислены новые глобальные диапазоны.")
        global_ranges = calculate_dicts(prepared_path, numeric_columns, group_cols)
        save_ranges(range_file, global_ranges)
        print_log(f"Глобальные диапазоны сохранены в {range_file}")


    normalize_ranges = global_ranges.copy()
    # Итерация по месяцам
    current_date = start_date_dt
    while current_date <= end_date_dt:
        # Определяем начало и конец текущего месяца
        month_start = current_date.strftime("%Y-%m-%d")
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Переход к следующему месяцу
        month_end = (next_month - timedelta(days=next_month.day)).strftime("%Y-%m-%d")

        print_log(f"обработка данных за {month_start} - {month_end}")    

        month_file = os.path.join(prepared_path, f'{current_date.strftime("%Y-%m")}.parquet')
        df = read_prepared(month_file)

            # Применяем циклические признаки для даты
        df = add_cyclic_features(df, "month", 1, 12)
        df = add_cyclic_features(df, "day_of_week", 1, 7)
        df = add_normalized_day_column(df, 2010, 2100)
        df = add_cyclic_features_with_days(df, "day_of_month", "month")

        df = normalize_data(df, numeric_columns, group_cols, normalize_ranges, global_ranges)

        print_log(f"сохранение данных за {month_start} - {month_end}")
        file_name = f"{month_start[:7]}.parquet"  # Берём только год-месяц из строки даты
        file_path = os.path.join(normalized_path, file_name)

        # Сохраняем DataFrame в файл
        df.to_parquet(file_path, index=False)  # index=False, чтобы не сохранять индексы
        print_log(f"сохранены данные {month_start} - {month_end}")

        current_date = next_month.replace(day=1)

        del df
        gc.collect()

    check_ranges_for_retraining(normalize_ranges, global_ranges, numeric_columns)

if __name__ == "__main__":
    main()
