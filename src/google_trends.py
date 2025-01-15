import pandas as pd
import os
import numpy as np
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
from pytrends.exceptions import TooManyRequestsError

last_log_time = time.time()
def print_log(message):
    global last_log_time
    
    # Получаем текущее время
    current_time = time.time()
    
    # Рассчитываем время с последнего лога
    time_diff = current_time - last_log_time
    last_log_time = current_time  # Обновляем время последнего лога
    
    print(f"[{int(time_diff)}s] {message}")  # Обновляем строку с временем

def read_existing_trends(file_path):
    """Читает файл trends.csv или возвращает пустой DataFrame."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path, parse_dates=['date'])
    return pd.DataFrame({
        'date': pd.Series(dtype='datetime64[ns]'),  # Пустой столбец дат
        'keywords': pd.Series(dtype='str'),        # Пустой столбец строк
    })

def determine_start_date(trends_data, keyword, fixed_start_date):
    """Определяет начальную дату для keyword."""
    if keyword in trends_data['keywords'].unique():
        max_date = trends_data[trends_data['keywords'] == keyword]['date'].max()
        return max_date
    return fixed_start_date

def fetch_trends(keywords_string, keywords, start_date, end_date, geo='KZ', max_retries=5, pause=15):
    """Получает данные трендов для keyword в заданный период с обработкой ошибок."""
    pytrends = TrendReq(hl='ru', tz=300)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

    retries = 0
    while retries < max_retries:
        try:
            pause_retry = pause * (1 + (retries/2))
            print_log(f"ожидание: {pause_retry} сек.")
            time.sleep(pause_retry)

            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
            trend_data = pytrends.interest_over_time()

            if trend_data.empty:
                return create_dataframe_with_dates_and_keywords(start_date, end_date, keywords)
            return trend_data
        except TooManyRequestsError as e:
            print_log(f"Ошибка: {e}")
            retries += 1
            if retries > max_retries:
                print(f"Достигнуто максимальное число попыток для ключевого слова: {keywords_string}")
                return pd.DataFrame(columns=['date', 'keywords', 'trend_value'])
            print(f"Слишком много запросов. Повтор через {pause} секунд (попытка {retries}/{max_retries})...")
 
        except Exception as e:
            print_log(f"Ошибка: {e}")

def create_dataframe_with_dates_and_keywords(start_date, end_date, keywords):
    # Генерируем даты
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Создаем датафрейм с датами
    df = pd.DataFrame({'date': date_range})
    
    # Добавляем числовые колонки
    for i in range(len(keywords)):
        df[str(i)] = 0
    
    return df
        
def update_trends_file(trend_data, file_path):
    if os.path.exists(file_path):
        # Читаем существующие данные
        existing_data = pd.read_csv(file_path)
        
        # Находим новые строки
        new_data = trend_data[~trend_data[['date', 'keywords']].isin(existing_data[['date', 'keywords']]).all(axis=1)]
        
        # Дописываем только новые данные
        if not new_data.empty:
            new_data.to_csv(file_path, mode='a', index=False, header=False)
    else:
        # Создаём файл, если он отсутствует
        trend_data.to_csv(file_path, index=False)

def create_groups_trends_file(excel_path, trends_path, groups_trends_path):
    trends = read_existing_trends(trends_path)
    groups = pd.read_excel(excel_path)

    # Выбираем нужные колонки из trends
    trends_subset = trends[['keywords', 'date', 'average_trend', 'trend_growth_rate']].rename(columns={'date': 'operDay'})
    
    # Левое соединение по ключу keywords
    groups_trends = groups.merge(trends_subset, on='keywords', how='left')
    
    # Сохранение в Parquet
    groups_trends.to_parquet(groups_trends_path, index=False)  

    print_log(f"данные записаны в {groups_trends_path}")  

def main():
    # Задаем параметры
    excel_path = "C:\\python_projects\\sales_forecast\\data\\groups.xlsx"
    trends_path = "C:\\python_projects\\sales_forecast\\data\\trends.csv"
    groups_trends_path = "C:\\python_projects\\sales_forecast\\data\\groups_trends.parquet"
    fixed_start_date = datetime(2024, 1, 1)  # Фиксированная начальная дата
    #end_date = datetime.now() - timedelta(days=1)  # До вчерашнего дня
    end_date = datetime(2024, 12, 31) 
    geo = 'KZ'

    pause = 15

    # Шаг 1: Чтение данных
    groups = pd.read_excel(excel_path)
    trends_data = read_existing_trends(trends_path)

    # Шаг 2: Уникальные keywords
    unique_keywords = groups['keywords'].unique()

    # Шаг 3: Итерация по keywords
    total_keywords = len(unique_keywords)
    for idx, keywords_string in enumerate(unique_keywords, start=1):
        start_date = determine_start_date(trends_data, keywords_string, fixed_start_date)

        print_log(f"{idx}/{total_keywords}: {keywords_string} from {start_date}")

        # Шаг 4: Запрос данных по полугодовым интервалам
        while start_date < end_date:
            next_end_date = min(start_date + timedelta(days=183), end_date)

            # массив не более чем из 5 элементов
            keywords = [item.strip() for item in keywords_string.split(",")][:5]
            trend_data = fetch_trends(keywords_string, keywords, start_date, next_end_date, geo, pause=pause)

            if not trend_data.empty:
                trend_data = trend_data.sort_values(by='date')
                trend_data['keywords'] = keywords_string

                if 'isPartial' in trend_data.columns:
                    trend_data = trend_data.drop(columns=['isPartial'])

                trend_data['average_trend'] = trend_data.select_dtypes(include=['number']).mean(axis=1)
                trend_data['trend_growth_rate'] = ((trend_data['average_trend'] - trend_data['average_trend'].shift(1)) / 100).fillna(0)
                trend_data = trend_data.reset_index()

                # Получаем индексы колонок из keyword
                keyword_columns = [col for col in trend_data.columns if col in keywords]

                # Переименовываем только эти колонки
                for i, col in enumerate(keyword_columns):
                    trend_data.rename(columns={col: str(i)}, inplace=True)

                trends_columns = [str(i) for i in range(5)]
                for col in trends_columns:
                    if col not in trend_data.columns:
                        trend_data[col] = np.nan

                trend_data = trend_data[['date', 'keywords', 'average_trend', 'trend_growth_rate'] + trends_columns]
                
                update_trends_file(trend_data, trends_path)
            start_date = next_end_date + timedelta(days=1)

    print_log(f"данные трендов получены")
    create_groups_trends_file(excel_path, trends_path, groups_trends_path)

if __name__ == "__main__":
    main()
