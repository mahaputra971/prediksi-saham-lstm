from integrations import get_issuer, get_emiten_id, insert_data_analyst, get_table_data
import yfinance as yf
from datetime import datetime, timedelta
import schedule
import time
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel

class StockPriceResponse(BaseModel):
    kode_emiten: str
    open: float
    high: float
    low: float
    close: float
    close_adj: float
    volume: int
    date: str

def add_data_engine(stock, stock_nama):
# Process each stock separately
    detail_emiten = get_table_data(stock, 'tb_detail_emiten')
    detail_emiten = [StockPriceResponse(**{**item, 'kode_emiten': stock}) for item in detail_emiten]
    
    for item in detail_emiten:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
    
    # Set a default value for latest_date
    latest_date = datetime.strptime("1900-01-01", '%Y-%m-%d')

    # If detail_emiten is not empty, find the maximum date
    if detail_emiten:
        latest_date = max(item.date for item in detail_emiten)
    
    if latest_date.date() < datetime.now().date():
        # print(f"tanggal terbaru di data : {latest_date.date()}")
        # print(f"tanggal sekarang : {datetime.now().date()}")
        stock_id = get_emiten_id(stock)
        print(stock_id)
        print(stock)
        print(stock_nama)
        data = yf.download(stock, start=latest_date.date() + timedelta(days=1), end=datetime.now().date())
        # print(data)
        # print(data.tail())
        
        df_copy = data.reset_index()
        df_copy['id_emiten'] = stock_id
        
        df_copy = df_copy.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'close_adj',
            'Volume': 'volume'
        })
        # Convert pandas Timestamp objects to datetime.datetime objects
        df_copy['date'] = df_copy['date'].apply(lambda x: x.to_pydatetime().strftime('%Y-%m-%d'))
        insert_data_analyst("tb_detail_emiten", df_copy) 

def job():
    start_time = time.time()
    # grab data
    stock_data, company_name = get_issuer()
    for stock, stock_nama in zip(stock_data, company_name):
        add_data_engine(stock, stock_nama)
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"The program took {elapsed_time} seconds to run.")

def main():
    # Schedule the job to run at 12:00 PM every day
    schedule.every().day.at("01:37").do(job)

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
    