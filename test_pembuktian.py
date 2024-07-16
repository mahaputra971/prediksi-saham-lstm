import pandas as pd
from sqlalchemy import create_engine, text
from integrations import get_trend_arrays, get_emiten_id

def validation_close_price(current_date, conn, status_ichimoku, array_1_waktu, close_price, id_stock, hari): 
    i = hari
    close_prices = []

    while len(close_prices) == 0 and i < 10000:
        for j in range(1, i):
            date = current_date + pd.Timedelta(days=j)
            price = conn.execute(text("""
                SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
            """), {'date': date, 'id_stock': id_stock}).scalar()
            if price is not None:
                close_prices.append(price)
        i += 1

    # validation close_prices should not be empty
    if len(close_prices) > 0:
        highest = max(close_prices)
        lowest = min(close_prices)
        if status_ichimoku == 1:
            array_1_waktu.append(1 if highest > close_price else 0)
        elif status_ichimoku == 0:
            array_1_waktu.append(1 if lowest < close_price else 0)
        else:
            array_1_waktu.append(1 if highest <= close_price or lowest >= close_price else 0)
    # print(f"136: {array_1_minggu}")
    
    return array_1_waktu

def pembuktian_ichimoku(stock, type):
    id_stock = get_emiten_id(stock)
    print(f"7: {id_stock}")
    
    # Create database connection
    engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')

    # Fetch the earliest date from tb_ichimoku_cloud for the given stock
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MIN(date) FROM tb_data_ichimoku_cloud WHERE id_emiten = :id_stock"), {'id_stock': id_stock})
        date_ichimoku = result.scalar()
    
    array_1_hari = []
    array_1_minggu = []
    array_1_bulan = []
    
    if not date_ichimoku:
        return array_1_hari, array_1_minggu, array_1_bulan
    
    # Loop through each day from the starting date to the last date
    current_date = pd.to_datetime(date_ichimoku)
    tgl_sekarang = current_date
    with engine.connect() as conn:
        while True:
            try:
                # Fetch data for current date
                query = text("""
                    SELECT tenkan_sen, kijun_sen FROM tb_data_ichimoku_cloud WHERE date = :date AND id_emiten = :id_stock
                """)
                ichimoku_data = conn.execute(query, {'date': current_date, 'id_stock': id_stock}).fetchone()
                # print(f"34: {ichimoku_data}")

                query = text("""
                    SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
                """)
                close_price = conn.execute(query, {'date': current_date, 'id_stock': id_stock}).scalar()
                # print(f"40: {close_price}")

                if not ichimoku_data or close_price is None:
                    current_date += pd.Timedelta(days=1)
                    continue
                
                # Calculate status_ichimoku
                if type == 'sen': 
                    tenkan_sen, kijun_sen = ichimoku_data
                    # print(f"49: {tenkan_sen}")
                    # print(f"50: {kijun_sen}")
                    
                    if close_price > tenkan_sen > kijun_sen:
                        status_ichimoku = 1
                    elif close_price < tenkan_sen < kijun_sen:
                        status_ichimoku = 0
                    elif close_price < tenkan_sen > kijun_sen:
                        status_ichimoku = 0.5
                    elif close_price > tenkan_sen < kijun_sen:
                        status_ichimoku = 1
                    elif close_price < tenkan_sen == kijun_sen:
                        status_ichimoku = 0
                    elif close_price > tenkan_sen == kijun_sen:
                        status_ichimoku = 1
                    elif close_price == tenkan_sen < kijun_sen:
                        status_ichimoku = 0.5
                    elif close_price == tenkan_sen > kijun_sen:
                        status_ichimoku = 0.5
                    elif close_price == tenkan_sen == kijun_sen:
                        status_ichimoku = 0.5
                    elif tenkan_sen < close_price < kijun_sen:
                        status_ichimoku = 0.5
                    elif tenkan_sen > close_price > kijun_sen:
                        status_ichimoku = 0.5
                    else:
                        status_ichimoku = 0.5
                        
                elif type == 'span':
                    senkou_span_a, senkou_span_b = ichimoku_data
                    # print(f"79: {senkou_span_a}")
                    # print(f"80: {senkou_span_b}")
                    if close_price > senkou_span_a and senkou_span_a > senkou_span_b:
                        status_ichimoku = 1
                    elif close_price < senkou_span_a and senkou_span_a < senkou_span_b:
                        status_ichimoku = 0
                    elif close_price < senkou_span_b and senkou_span_a > senkou_span_b:
                        status_ichimoku = 0
                    elif close_price > senkou_span_b and senkou_span_a < senkou_span_b:
                        status_ichimoku = 1
                    elif senkou_span_b < close_price < senkou_span_a and senkou_span_a > senkou_span_b:
                        status_ichimoku = 1
                    elif senkou_span_b < close_price < senkou_span_a and senkou_span_a < senkou_span_b:
                        status_ichimoku = 0
                    else:
                        status_ichimoku = 0.5
                        
                else:
                    print("Invalid type, please input either 'sen' or 'span'")
                    break
                        
                # print(f"100: {status_ichimoku}")

                # Fetch next day close price
                next_day = current_date + pd.Timedelta(days=1)
                print(f"104: {next_day}")
                next_day_close = conn.execute(text("""
                    SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
                """), {'date': next_day, 'id_stock': id_stock}).scalar()
                # print(f"108: {next_day_close}")

                if next_day_close is not None:
                    compare_1_hari = 1 if close_price < next_day_close else 0 if close_price > next_day_close else 0.5
                    array_1_hari.append(1 if compare_1_hari == status_ichimoku else 0)
                # print(f"113: {array_1_hari}")

                # Fetch one week close prices
                array_1_minggu = validation_close_price(current_date, conn, status_ichimoku, array_1_minggu, close_price, id_stock, 8)

                # Fetch one month close prices
                array_1_bulan = validation_close_price(current_date, conn, status_ichimoku, array_1_bulan, close_price, id_stock, 31)

                # Move to the next date
                current_date += pd.Timedelta(days=1)
                # print(f"163: {current_date}")

                # Check if current_date is the last date in tb_data_ichimoku_cloud or tb_detail_emiten
                last_date_ichimoku = conn.execute(text("SELECT MAX(date) FROM tb_data_ichimoku_cloud WHERE id_emiten = :id_stock"), {'id_stock': id_stock}).scalar()
                last_date_detail = conn.execute(text("SELECT MAX(date) FROM tb_detail_emiten WHERE id_emiten = :id_stock"), {'id_stock': id_stock}).scalar()
                
                if current_date.date() > last_date_ichimoku or current_date.date() > last_date_detail:
                    break

            except Exception as e:
                print(f"Error occurred: {e}")
                break
    
    return array_1_hari, array_1_minggu, array_1_bulan, tgl_sekarang

def main():
    stock = 'ZATA.JK'
    
    #SEN
    array_1_hari_sen, array_1_minggu_sen, array_1_bulan_sen,tgl_sekarang = pembuktian_ichimoku(stock, 'sen')
    print(array_1_hari_sen)
    print(array_1_minggu_sen)
    print(array_1_bulan_sen)
    
    percent_1_hari_sen = pd.Series(array_1_hari_sen).mean() * 100
    percent_1_minggu_sen = pd.Series(array_1_minggu_sen).mean() * 100
    percent_1_bulan_sen = pd.Series(array_1_bulan_sen).mean() * 100
    
    #SPAN
    array_1_hari_span, array_1_minggu_span, array_1_bulan_span,tgl_sekarang = pembuktian_ichimoku(stock, 'span')
    print(array_1_hari_span)
    print(array_1_minggu_span)
    print(array_1_bulan_span)
    
    percent_1_hari_span = pd.Series(array_1_hari_span).mean() * 100
    percent_1_minggu_span = pd.Series(array_1_minggu_span).mean() * 100
    percent_1_bulan_span = pd.Series(array_1_bulan_span).mean() * 100
    
    #FINAL RESULT
    print(f"Percentage of True (SEN_FACTOR) in array_1_hari: {percent_1_hari_sen}%")
    print(f"Percentage of True (SEN_FACTOR) in array_1_minggu: {percent_1_minggu_sen}%")
    print(f"Percentage of True (SEN_FACTOR) in array_1_bulan: {percent_1_bulan_sen}%")

    print(f"\nPercentage of True (span_FACTOR) in array_1_hari: {percent_1_hari_span}%")
    print(f"Percentage of True (span_FACTOR) in array_1_minggu: {percent_1_minggu_span}%")
    print(f"Percentage of True (span_FACTOR) in array_1_bulan: {percent_1_bulan_span}%")
    
    print(f"\nLenght of array_1_hari_sen: {len(array_1_hari_sen)}")
    print(f"Lenght of array_1_minggu_sen: {len(array_1_minggu_sen)}")
    print(f"Lenght of array_1_bulan_sen: {len(array_1_bulan_sen)}")
    
    print(f"\nCurrent Date : {tgl_sekarang}")
    
if __name__ == '__main__':
    main()
