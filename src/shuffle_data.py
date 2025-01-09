import os
import pandas as pd
import numpy as np
import shutil
import time
from datetime import datetime

# Исходный и целевой каталоги
input_dir = "C:\\python_projects\\sales_forecast\\data\\normalized"
temp_dir = "C:\\python_projects\\sales_forecast\\data\\temp"
output_dir = "C:\\python_projects\\sales_forecast\\data\\shuffled"

last_log_time = time.time()

def print_log(message):
    global last_log_time
    
    # Получаем текущее время
    current_time = time.time()
    
    # Рассчитываем время с последнего лога
    time_diff = current_time - last_log_time
    last_log_time = current_time  # Обновляем время последнего лога
    
    print(f"[{int(time_diff)}s] {message}")  # Обновляем строку с временем

def clear_directory(directory):

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  # Удаление файлов и символических ссылок
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Удаление папок

    print_log("каталоги очищены")

def shuffle_temp(parquet_files):

    num_files = len(parquet_files)

    # Прочитать каждый файл и записать в целевой каталог
    for file in parquet_files:
        print_log(f"обработка файла {file}")

        input_path = os.path.join(input_dir, file)
        
        # Чтение и запись
        df = pd.read_parquet(input_path)
        df = df.sample(frac=1).reset_index(drop=True)

        parts = np.array_split(df, num_files)
        
        for idx, df_part in enumerate(parts, start=1):
            temp_dir_for_file = os.path.join(temp_dir, str(idx))
            os.makedirs(temp_dir_for_file, exist_ok=True)

            file_name = os.path.splitext(os.path.basename(file))[0]
            temp_path_for_file = os.path.join(temp_dir_for_file, file_name + ".parquet")
            df_part = df_part.reset_index(drop=True)  # Сбрасываем индекс
            df_part = df_part.reindex(columns=df.columns)  # Обеспечиваем одинаковую структуру колонок
            df_part.to_parquet(temp_path_for_file, index=False, compression="snappy")

        print_log(f"обработан файл {file}")

def get_files_by_date_range(directory, start_date, end_date):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            file_date = datetime.strptime(filename, "%Y-%m.parquet")
            if start_date <= file_date <= end_date:
                files.append(os.path.join(directory, filename))
    print_log(f"выбраны файлы из {directory}")
    return files

def process_directories():

    # Перебираем каталоги в temp_dir
    for folder in os.listdir(temp_dir):
        folder_path = os.path.join(temp_dir, folder)

        print_log(f"начало обработки {folder_path}")
        if os.path.isdir(folder_path):  # Проверяем, что это каталог
            # Собираем все файлы Parquet внутри каталога
            parquet_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".parquet")
            ]

            # Объединяем все Parquet-файлы в один DataFrame
            df_list = [pd.read_parquet(file) for file in parquet_files]
            combined_df = pd.concat(df_list, ignore_index=True)

            # Перемешиваем строки
            shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

            # Сохраняем в output_dir с именем как у каталога
            output_file = os.path.join(output_dir, f"{folder}.parquet")
            shuffled_df.to_parquet(output_file, index=False)

            print_log(f"обработаны файлы из {folder_path}")

def main():

    os.makedirs(output_dir, exist_ok=True)    
    os.makedirs(temp_dir, exist_ok=True)    

    clear_directory(output_dir)
    clear_directory(temp_dir)

    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 3, 31)
    files_to_load = get_files_by_date_range(input_dir, start_date, end_date)

    shuffle_temp(files_to_load)

    process_directories()

if __name__ == "__main__":
    main()