from pytrends.request import TrendReq
from datetime import datetime, timedelta

# Создание объекта с прокси
#pytrends = TrendReq(hl='ru', tz=300)

#Создаем объект API
#pytrends = TrendReq(hl='ru', tz=300)

# Задаем параметры
#keywords = ['лимонад']  # Замените на нужные ключевые слова
#geo = 'KZ'  # Регион 
#timeframe = '2024-01-01 2024-06-29'  # Период

#Запрос популярности для группы ключевых слов
#pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')

#Получаем данные популярности
#trend_data = pytrends.interest_over_time()
#print(trend_data)

keyword = ['лимонад', 'газировка', 'буратино']
geo='KZ'
pytrends = TrendReq(hl='ru', tz=300)
start_date = datetime(2024, 1, 1) 
end_date = start_date + timedelta(days=200)

timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

pytrends.build_payload(keyword, cat=0, timeframe=timeframe, geo=geo, gprop='')
trend_data = pytrends.interest_over_time()

# Получаем индексы колонок из keyword
keyword_columns = [col for col in trend_data.columns if col in keyword]

# Переименовываем только эти колонки
for i, col in enumerate(keyword_columns):
    trend_data.rename(columns={col: str(i)}, inplace=True)
    
print(trend_data)