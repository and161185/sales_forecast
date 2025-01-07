import os
import time
import pandas as pd
from clickhouse_connect import create_client
from datetime import datetime, timedelta
from calendar import monthrange

last_log_time = time.time()

def print_log(message):
    global last_log_time
    
    # Получаем текущее время
    current_time = time.time()
    
    # Рассчитываем время с последнего лога
    time_diff = current_time - last_log_time
    last_log_time = current_time  # Обновляем время последнего лога
    
    print(f"[{int(time_diff)}s] {message}")  # Обновляем строку с временем

def fetch_and_save_sales_data(start_date, end_date, output_dir):
    """
    Выполняет запрос к ClickHouse для указанного диапазона дат и сохраняет результаты в общий CSV-файл.
    
    :param start_date: Начало диапазона (включительно), формат 'YYYY-MM-DD'.
    :param end_date: Конец диапазона (не включительно), формат 'YYYY-MM-DD'.
    :param output_file: Путь к файлу CSV для сохранения результата.
    """
    # Создаем клиента
    client = create_client(
        host='192.168.1.75', 
        port=8123,
        username='admin', 
        password='admin', 
        database='SetRetailAPI'
    )

    # Запрос с подстановкой диапазона дат
    query = f"""
       SELECT 
        prch.operDay as operDay,
        sh.code as shop, 
        g.code as goodsCode1c, 
        round(SUM(pos.amount) / SUM(pos.count), 2) as avg_price,
        round(SUM(pos.count), 2) AS count, 
        MAX(ldp.action_type) as action_type,
        SUM(pos.amount) AS amount, 
        SUM(pos.discountValue) AS discount_value,
        SUM(CASE WHEN ldp.action_type IS NULL THEN
        	0
        ELSE
        	pos.amount
        END) as action_amount,
        round(SUM(CASE WHEN ldp.action_type IS NULL THEN
        	0
        ELSE
        	pos.count
        END), 2) as action_count,
        round(SUM(CASE WHEN ldp.action_type IS NULL THEN
        	0
        ELSE
        	pos.amount
        END) /
        SUM(CASE WHEN ldp.action_type IS NULL THEN
        	0
        ELSE
        	pos.count
        END), 2) as action_avg_price
    FROM 
        (SELECT UID_PURCHASE, toDate(operDay) AS operDay, shop, shift, cash, number 
         FROM SetRetailAPI.purchases 
         WHERE amount <> 0 
         AND operDay >= '{start_date}' 
         AND operDay < '{end_date}') prch
    INNER JOIN SetRetailAPI.Shops as sh
        ON prch.shop = sh.number         
    INNER JOIN 
        (SELECT UID_PURCHASE, 
            if(position(goodsCode, '-') > 0, arrayElement(splitByString('-', goodsCode), 1), goodsCode) as goodsCode, 
        	count * if(position(goodsCode, '-') > 0, toInt32(arrayElement(splitByString('-', goodsCode), 2)), 1) as count,
        	amount, cost, discountValue, order
         FROM SetRetailAPI.positions
         WHERE dateCommit >= '{start_date}'
         AND dateCommit < '{end_date}' 
         AND count > 0) pos
        ON prch.UID_PURCHASE = pos.UID_PURCHASE
    INNER JOIN SetRetailAPI.Goods as g
        ON pos.goodsCode = g.goodsCode        
    LEFT JOIN 
        (SELECT cash_number, shift_number, purchase_number, shop_number, iddisc 
         FROM SetRetailAPI.loy_transactions
         WHERE transaction_time >= '{start_date}'
         AND transaction_time < '{end_date}') lt
        ON (lt.cash_number = prch.cash) AND (lt.shift_number = prch.shift) AND (lt.purchase_number = prch.number) AND (lt.shop_number = prch.shop)
    LEFT JOIN SetRetailAPI.loy_discount_position as ldp
        ON (ldp.transaction_id = lt.iddisc) AND (ldp.shop_number = lt.shop_number) AND (ldp.position_order = pos.order)    
    GROUP BY prch.operDay, sh.code, g.code
	ORDER BY g.code, sh.code, prch.operDay
    """

    # Выполнение запроса и сохранение результата
    try:
        results = client.query(query).result_rows
        headers = client.query(query).column_names

        # Преобразуем данные в DataFrame
        df = pd.DataFrame(results, columns=headers)

        # Генерируем имя файла
        month_name = start_date[:7]
        output_file = os.path.join(output_dir, f"{month_name}.parquet")

        # Сохраняем в формате Parquet
        df.to_parquet(output_file, engine='pyarrow', index=False)

        print_log(f"Данные за период {start_date} - {end_date} успешно добавлены в {output_file}")

    except Exception as e:
        print_log(f"Ошибка выполнения запроса или сохранения данных: {e}")



if __name__ == "__main__":
    # Начальная дата
    start_date = datetime.strptime('2024-01-01', '%Y-%m-%d')
    
    # Конечная дата (сегодня)
    end_date = datetime.now()

    #start_date = datetime.strptime('2024-11-01', '%Y-%m-%d')
    #end_date = datetime.strptime('2024-11-30', '%Y-%m-%d')

    # Путь для сохранения файлов
    retail_sales_dir = os.path.join(os.getcwd(), 'data', 'retail_sales_data')
    os.makedirs(retail_sales_dir, exist_ok=True)

    # Итерация по месяцам
    current_date = start_date
    while current_date < end_date:
        # Находим последний день текущего месяца
        last_day_of_month = monthrange(current_date.year, current_date.month)[1]
        month_start = current_date.strftime('%Y-%m-%d')  # Начало месяца
        month_end = current_date.replace(day=last_day_of_month).strftime('%Y-%m-%d')  # Конец месяца

        # Выполняем запрос и сохраняем результат
        fetch_and_save_sales_data(month_start, month_end, retail_sales_dir)

        # Переходим к следующему месяцу
        next_month = current_date.replace(day=28) + timedelta(days=4)  # Перепрыгиваем на следующий месяц
        current_date = next_month.replace(day=1)

    print_log(f"Данные успешно собраны и сохранены в {retail_sales_dir}")
