import os
import pandas as pd
import numpy as np

def process_parquet_files(directory):
    # Перебираем все файлы в указанном каталоге
    for filename in os.listdir(directory):
        if filename.endswith('.parquet'):
            filepath = os.path.join(directory, filename)
            
            # Читаем файл Parquet
            data = pd.read_parquet(filepath)
            
            # Добавляем новый столбец
            data['is_sell'] = np.where(data['allSalesCount'] > 0, 1, 0)
            
            # Сохраняем изменения в том же файле
            data.to_parquet(filepath, index=False)
            print(f"Обработан файл: {filename}")

# Пример вызова
directory_path = "C:\\python_projects\\sales_forecast\\data\\shuffled"
process_parquet_files(directory_path)
