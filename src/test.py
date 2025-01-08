import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import gc
import time

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


leftovers_path = "C:\\python_projects\\sales_forecast\\data\\leftovers\\2024-01.csv"
retail_sales_path = "C:\\python_projects\\sales_forecast\\data\\retail_sales_data\\2024-01.parquet"

data = pd.read_csv(leftovers_path, parse_dates=['operDay'], dtype = dtype_leftovers, engine='pyarrow') 
data['operDay'] = pd.to_datetime(data['operDay'])
data = data[(data['goodsCode1c'] == '1-00001824') & (data['shop'] == 2)]


retail_sales = pd.read_parquet(retail_sales_path)
retail_sales['operDay'] = pd.to_datetime(retail_sales['operDay'])

merged_data = pd.merge(
        data,
        retail_sales,
        on=['operDay', 'shop', 'goodsCode1c'],
        how='left'
    )

merged_data['avg_price'] = merged_data['avg_price'].fillna(0)
merged_data.loc[merged_data['avg_price'] == 0, 'avg_price'] = merged_data['price']

if not data[['shop', 'goodsCode', 'operDay']].sort_values(['shop', 'goodsCode', 'operDay']).equals(data):
    data = data.sort_values(by=['shop', 'goodsCode', 'operDay'])

grouped = merged_data.groupby(['shop', 'goodsCode'], group_keys=False, observed=True)

#скользящее среднее по количеству не можем считать от текущей строки, т.к. на сегодня продажи неизвестны, делаем shift(1)
data['count_ma_7'] = grouped['count'].shift(1).rolling(7, min_periods=1).mean().round(3)
data['count_ma_30'] = grouped['count'].shift(1).rolling(30, min_periods=1).mean().round(3)
data['allSalesCount_ma_7'] = grouped['allSalesCount'].shift(1).rolling(7, min_periods=1).mean().round(3)
data['allSalesCount_ma_30'] = grouped['allSalesCount'].shift(1).rolling(30, min_periods=1).mean().round(3)
