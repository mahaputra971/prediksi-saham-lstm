from pandas_datareader import data as pdr
import yfinance as yf
import datetime
import json
# import mongo
import matplotlib.pyplot as plt
# from sql import insert_data_analyst
from . import insert_data_analyst, get_emiten_id
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression 


yf.pdr_override()

# def add_pic_summary(stock_param):
#     img_path = f'ichimokuproject/image/plot_IC_{stock_param}.png'
#     if os.path.exists(img_path):
#         with open(img_path, 'rb') as img_file:
#             return img_file.read()
#     else:
#         return pd.NA

def get_trend(tenkan_sen): 
    trends = []
    for i in range(len(tenkan_sen) - 1): 
        if tenkan_sen[i] == tenkan_sen[i+1]:
            trends.append(0)
        elif tenkan_sen[i] < tenkan_sen[i+1]:
            trends.append(+1)
        else: 
            trends.append(-1)
    return trends


def get_tenkan_sen_status(trends):
    sum = 0 # summary for each trend 
    for trend in trends:
        sum += trend
    
    # Determine the trend status
    tenkan_sen_status = str
    if sum > 0:
        tenkan_sen_status = "The Tenkan-Sen is in an uptrend."
    elif sum < 0:
        tenkan_sen_status = "The Tenkan-Sen is in a downtrend."
    else:
        tenkan_sen_status = "The Tenkan-Sen is moving sideways."

    return tenkan_sen_status

def get_kijun_sen_status(last_close, last_kijun_sen):
    kijun_sen_status = ""
    if last_close > last_kijun_sen:
        kijun_sen_status = "The kijun sen is in an upward trend."
    elif last_close < last_kijun_sen:
        kijun_sen_status = "The kijun sen is in a downward trend."
    else:
        kijun_sen_status = "The kijun sen is moving sideways."
    return  kijun_sen_status

def get_senkou_span_status(last_close, last_senkou_span_a, last_senkou_span_b):
    senkou_span_status = ""
    # Determine the market trend and potential price movements based on the position of the closing price relative to the Senkou Span A and B lines
    if last_close > last_senkou_span_a and last_senkou_span_a > last_senkou_span_b:
        senkou_span_status = "Senkou_Span Uptrend"
    elif last_close < last_senkou_span_a and last_senkou_span_a < last_senkou_span_b:
        senkou_span_status = "Senkou_Span Downtrend"
    elif last_close < last_senkou_span_b and last_senkou_span_a > last_senkou_span_b:
        senkou_span_status = "Senkou_Span Will Dump"
    elif last_close > last_senkou_span_b and last_senkou_span_a < last_senkou_span_b:
        senkou_span_status = "Senkou_Span Will Pump"
    elif last_senkou_span_b < last_close < last_senkou_span_a and last_senkou_span_a > last_senkou_span_b:
        senkou_span_status = "Senkou_Span Uptrend and Will Bounce Up"
    elif last_senkou_span_b < last_close < last_senkou_span_a and last_senkou_span_a < last_senkou_span_b:
        senkou_span_status = "Senkou_Span Downtrend and Will Bounce Down"
    else:
        senkou_span_status = "Senkou_Span Unknown"
    return senkou_span_status


def ichimoku_project(stock):
    try:
        # Define constants
        DAYS_9 = 9
        DAYS_26 = 26
        DAYS_52 = 52

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)

        # Fetch data and handle potential errors
        try:
            df = pdr.get_data_yahoo(stock, start=start_date, end=end_date.strftime("%Y-%m-%d"))
            print("\n\nFetched data for", stock)
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return None
        
        if df is None:
            print("Data is None after fetching")
        else:
            print("Data fetched successfully")

        high_9_days = df.High.rolling(DAYS_9).max() 
        low_9_days = df.High.rolling(DAYS_9).min()   
        high_26_days = df.High.rolling(DAYS_26).max()
        low_26_days = df.High.rolling(DAYS_26).min()
        high_52_days = df.High.rolling(DAYS_52).max()
        low_52_days = df.High.rolling(DAYS_52).min()
        
        df['tenkan_sen'] = (high_9_days + low_9_days) / 2
        df['kijun_sen'] = (high_26_days + low_26_days) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(DAYS_26) 
        df['senkou_span_b'] = ((high_52_days + low_52_days) / 2).shift(DAYS_26)
        df['chikou'] = df.Close.shift(-DAYS_26)

        plt.figure(figsize=(16,6))
        plt.plot(df.index, df['tenkan_sen'], lw=0.7, color='purple')
        plt.plot(df.index, df['kijun_sen'], lw=0.7, color='yellow')
        plt.plot(df.index, df['chikou'], lw=0.7, color='grey')
        plt.plot(df.index, df['Close'], lw=0.7, color='blue')
        plt.title("Ichimoku Saham : " + str(stock))
        plt.ylabel("Harga")

        kumo = df['Adj Close'].plot(lw=0.7, color='red')
        kumo.fill_between(df.index, df.senkou_span_a, df.senkou_span_b, where= df.senkou_span_a >= df.senkou_span_b, color='lightgreen')
        kumo.fill_between(df.index, df.senkou_span_a, df.senkou_span_b, where= df.senkou_span_a < df.senkou_span_b, color='lightcoral')
        plt.grid()  
        plt.savefig(f'ichimokuproject/image/plot_IC_{stock}.png')
        plt.clf()  # Clear the current figure to start a new plot for the next asset

        # convert dataframe into array of json
        all_ichimoku_data = json.loads(df.reset_index().to_json(orient='records'))
        

        # perform insert data to mongoDB

		# return current_ichimoku_data and all the array ichimoku data from the data from this is an 2D array
        return all_ichimoku_data
    except Exception as e:
        return f"An error occurred while processing data: {e}"

def get_all_ichimoku_project_controller(stock):
    try:
        # Define constants
        DAYS_9 = 9
        DAYS_26 = 26
        DAYS_52 = 52

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)

        # Fetch data and handle potential errors
        try:
            df = pdr.get_data_yahoo(stock, start=start_date, end=end_date.strftime("%Y-%m-%d"))
            print("\n\nFetched data for", stock)
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return None
        
        if df is None:
            print("Data is None after fetching")
        else:
            print("Data fetched successfully")

        high_9_days = df.High.rolling(DAYS_9).max() 
        low_9_days = df.High.rolling(DAYS_9).min()   
        high_26_days = df.High.rolling(DAYS_26).max()
        low_26_days = df.High.rolling(DAYS_26).min()
        high_52_days = df.High.rolling(DAYS_52).max()
        low_52_days = df.High.rolling(DAYS_52).min()
        
        df['tenkan_sen'] = (high_9_days + low_9_days) / 2
        df['kijun_sen'] = (high_26_days + low_26_days) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(DAYS_26) 
        df['senkou_span_b'] = ((high_52_days + low_52_days) / 2).shift(DAYS_26)
        df['chikou'] = df.Close.shift(-DAYS_26)

        plt.figure(figsize=(16,6))
        plt.plot(df.index, df['tenkan_sen'], lw=0.7, color='purple')
        plt.plot(df.index, df['kijun_sen'], lw=0.7, color='yellow')
        plt.plot(df.index, df['chikou'], lw=0.7, color='grey')
        plt.plot(df.index, df['Close'], lw=0.7, color='blue')
        plt.title("Ichimoku Saham : " + str(stock))
        plt.ylabel("Harga")

        kumo = df['Adj Close'].plot(lw=0.7, color='red')
        kumo.fill_between(df.index, df.senkou_span_a, df.senkou_span_b, where= df.senkou_span_a >= df.senkou_span_b, color='lightgreen')
        kumo.fill_between(df.index, df.senkou_span_a, df.senkou_span_b, where= df.senkou_span_a < df.senkou_span_b, color='lightcoral')
        plt.grid()  
        plt.savefig(f'ichimokuproject/image/plot_IC_{stock}.png')
        plt.clf()  # Clear the current figure to start a new plot for the next asset

        # convert dataframe into array of json
        all_ichimoku_data = json.loads(df.reset_index().to_json(orient='records'))

        # perform insert data to mongoDB

		# return current_ichimoku_data and all the array ichimoku data from the data from this is an 2D array
        return all_ichimoku_data
    except Exception as e:
        return f"An error occurred while processing data: {e}"

def get_current_ichimoku_project_controller(stock):
    try:
        # Define constants
        DAYS_9 = 9
        DAYS_26 = 26
        DAYS_52 = 52

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)

        # Fetch data and handle potential errors
        try:
            df = pdr.get_data_yahoo(stock, start=start_date, end=end_date.strftime("%Y-%m-%d"))
            print("\n\nFetched data for", stock)
        except Exception as e:
            print(f"An error occurred while fetching data: {e}")
            return None
        
        if df is None:
            print("Data is None after fetching")
        else:
            print("Data fetched successfully")

        high_9_days = df.High.rolling(DAYS_9).max() 
        low_9_days = df.High.rolling(DAYS_9).min()   
        high_26_days = df.High.rolling(DAYS_26).max()
        low_26_days = df.High.rolling(DAYS_26).min()
        high_52_days = df.High.rolling(DAYS_52).max()
        low_52_days = df.High.rolling(DAYS_52).min()
        
        df['tenkan_sen'] = (high_9_days + low_9_days) / 2
        df['kijun_sen'] = (high_26_days + low_26_days) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(DAYS_26) 
        df['senkou_span_b'] = ((high_52_days + low_52_days) / 2).shift(DAYS_26)
        df['chikou'] = df.Close.shift(-DAYS_26)

        # Get the last price and the previous price
        #last_price = data['Close'].iloc[-1]
        #prev_price = data['Close'].iloc[-2]
        
        # Get the last Kumo cloud components
        last_senkou_span_a = df['senkou_span_a'].iloc[-1]
        last_senkou_span_b = df['senkou_span_b'].iloc[-1]

        # Get the last Kijun-Sen and Tenkan-Sen
        last_kijun_sen = df['kijun_sen'].iloc[-1]
        last_tenkan_sen = df['tenkan_sen'].iloc[-1]

        ####################################### SEKAN_SEN FACTOR
        tenkan_sen = df['tenkan_sen']
        trends = get_trend(tenkan_sen)

        tenkan_sen_status = get_tenkan_sen_status(trends)
        ######################################## KIJUN_SEN FACTOR
        # Get the last closing price and the last Kijun-Sen value
        last_close = df['Close'].iloc[-1]
        last_kijun_sen = df['kijun_sen'].iloc[-1]

        kijun_sen_status = get_kijun_sen_status(last_close, last_kijun_sen)
            
        ######################################## SENKOU_SEN (KUMO) FACTOR
        # Get the last closing price and the last Senkou Span A and B values
        last_close = df['Close'].iloc[-1]
        last_senkou_span_a = df['senkou_span_a'].iloc[-1]
        last_senkou_span_b = df['senkou_span_b'].iloc[-1]
        
        senkou_span_status = get_senkou_span_status(last_close, last_senkou_span_a, last_senkou_span_b)

		# assign variable for currenty day ichimoku value
        close_price= df['Close'].iloc[-1]
        tenkan_sen= df['tenkan_sen'].iloc[-1]
        kijun_sen= df['kijun_sen'].iloc[-1]
        senkou_span_a= df['senkou_span_a'].iloc[-1] 
        senkou_span_b= df['senkou_span_b'].iloc[-1] 
        date=datetime.datetime.now()  
        currentTime = date.strftime("%d-%m-%Y %H:%M:%S")
        date = currentTime

		# create object for currenty day ichimoku value
        current_ichimoku_data = {
            "kode_emiten": stock,
            "close_price": close_price,
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen, 
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "date": date,
            "tenkan_sen_status": tenkan_sen_status,
            "kijun_sen_status": kijun_sen_status,
            "senkou_span_status": senkou_span_status
        }

		# return current_ichimoku_data
        return current_ichimoku_data
    except Exception as e:
        return f"An error occurred while processing data: {e}"
    
def interpret_sen_status(last_close, last_tenkan_sen, last_kijun_sen):
    if last_close > last_tenkan_sen > last_kijun_sen:
        status_sen = "Pasar Bullish"
    elif last_close < last_tenkan_sen < last_kijun_sen:
        status_sen = "Pasar Bearish"
    elif last_close < last_tenkan_sen > last_kijun_sen:
        status_sen = "Pasar Tidak Stabil, Potensi Reversal"
    elif last_close > last_tenkan_sen < last_kijun_sen:
        status_sen = "Pasar dalam Transisi, Potensi Kenaikan"
    elif last_close < last_tenkan_sen == last_kijun_sen:
        status_sen = "Pasar Bearish Konsolidasi, Potensi Kelanjutan Penurunan"
    elif last_close > last_tenkan_sen == last_kijun_sen:
        status_sen = "Pasar Bullish Konsolidasi, Potensi Kelanjutan Kenaikan"
    elif last_close == last_tenkan_sen < last_kijun_sen:
        status_sen = "Pasar Tidak Stabil, Potensi Reversal"
    elif last_close == last_tenkan_sen > last_kijun_sen:
        status_sen = "Pasar Tidak Stabil, Potensi Reversal"
    elif last_close == last_tenkan_sen == last_kijun_sen:
        status_sen = "Pasar Sideways, Konsolidasi"
    elif last_tenkan_sen < last_close < last_kijun_sen:
        status_sen = "Pasar dalam Transisi, Ketidakpastian"
    elif last_tenkan_sen > last_close > last_kijun_sen:
        status_sen = "Pasar dalam Transisi, Ketidakpastian"
    else:
        status_sen = "Kondisi Tidak Terdefinisi"
    
    return status_sen

def ichimoku_sql(stock):
    print("ichimoku_project function called")
    
    # Define constants
    DAYS_9 = 9
    DAYS_26 = 26
    DAYS_52 = 52

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)

    # Fetch data and handle potential errors
    try:
        data = pdr.get_data_yahoo(stock, start=start_date, end=end_date.strftime("%Y-%m-%d"))
        print("\n\nFetched data for", stock)
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return None
    
    # Calculate 'n' as the length of the DataFrame
    n = len(data)
    
    if data is None:
        print("Data is None after fetching")
    else:
        print("Data fetched successfully")

    high_9_days = data.High.rolling(DAYS_9).max() 
    low_9_days = data.High.rolling(DAYS_9).min()   
    high_26_days = data.High.rolling(DAYS_26).max()
    low_26_days = data.High.rolling(DAYS_26).min()
    high_52_days = data.High.rolling(DAYS_52).max()
    low_52_days = data.High.rolling(DAYS_52).min()
    
    data['tenkan_sen'] = (high_9_days + low_9_days) / 2
    data['kijun_sen'] = (high_26_days + low_26_days) / 2
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(DAYS_26) 
    data['senkou_span_b'] = ((high_52_days + low_52_days) / 2).shift(DAYS_26)
    data['chikou'] = data.Close.shift(-DAYS_26)
    data = data.iloc[DAYS_26:]
    print(data.dtypes)
    data = data.dropna()
    is_finite = data.applymap(np.isfinite)
    print(is_finite)
    has_na = data.isna().any()
    print(has_na)

    plt.figure(figsize=(16,6))
    plt.plot(data.index, data['tenkan_sen'], lw=0.7, color='purple')
    plt.plot(data.index, data['kijun_sen'], lw=0.7, color='yellow')
    plt.plot(data.index, data['chikou'], lw=0.7, color='grey')
    plt.plot(data.index, data['Close'], lw=0.7, color='blue')
    plt.title("Ichimoku Saham : " + str(stock))
    plt.ylabel("Harga")
    kumo = data['Adj Close'].plot(lw=0.7, color='red')
    kumo.fill_between(data.index, data.senkou_span_a, data.senkou_span_b, where= data.senkou_span_a >= data.senkou_span_b, color='lightgreen')
    kumo.fill_between(data.index, data.senkou_span_a, data.senkou_span_b, where= data.senkou_span_a < data.senkou_span_b, color='lightcoral')
    plt.grid()  
    plt.savefig(f'picture/ichimoku/{stock}.png')
    # plt.savefig(f'ichimokuproject/image/plot_IC_{stock}.png')
    plt.clf()  # Clear the current figure to start a new plot for the next asset
    
    # print(data)

    print ("\nKode Emiten : ", stock)
    print("Close Price:", data['Close'].iloc[-1])
    print("tenkan_sen:", data['tenkan_sen'].iloc[-1])
    print("kijun_sen:", data['kijun_sen'].iloc[-1])
    print("senkou_span_a:", data['senkou_span_a'].iloc[-1])
    print("senkou_span_b:", data['senkou_span_b'].iloc[-1])
    print("chikou:", data['chikou'].iloc[-1])
    

    # Get the last Kumo cloud components
    last_senkou_span_a = data['senkou_span_a'].iloc[-1]
    last_senkou_span_b = data['senkou_span_b'].iloc[-1]

    # Get the last Kijun-Sen
    last_kijun_sen = data['kijun_sen'].iloc[-1]

    ####################################### TEKAN_SEN FACTOR
    # tenkan_sen = data['tenkan_sen']
    # trends = get_trend(tenkan_sen)
    # tenkan_sen_status = get_tenkan_sen_status(trends)
        
    ######################################## KIJUN_SEN FACTOR
    # Get the last closing price and the last Kijun-Sen value
    last_close = data['Close'].iloc[-1]
    last_kijun_sen = data['kijun_sen'].iloc[-1]

    # Determine the trend based on the position of the closing price relative to the Kijun-Sen line
    # if last_close > last_kijun_sen:
    #     kijun_sen_status = "The market is in an upward trend."
    # elif last_close < last_kijun_sen:
    #     kijun_sen_status = "The market is in a downward trend."
    # else:
    #     kijun_sen_status = "The market is moving sideways."
    # print(kijun_sen_status)
    
    sen_status = interpret_sen_status(last_close, last_kijun_sen, last_kijun_sen)
    
    ######################################## SENKOU_SEN (KUMO) FACTOR
    # Get the last closing price and the last Senkou Span A and B values
    last_close = data['Close'].iloc[-1]
    last_senkou_span_a = data['senkou_span_a'].iloc[-1]
    last_senkou_span_b = data['senkou_span_b'].iloc[-1]

    # Determine the market trend and potential price movements based on the position of the closing price relative to the Senkou Span A and B lines
    span_status = get_senkou_span_status(last_close, last_senkou_span_a, last_senkou_span_b)
    print(span_status)
    
    # Reset the index to move 'Date' from index to columns
    data_reset = data.reset_index()

    # Create a new DataFrame with only the desired columns
    data_new = data_reset[['Date', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']]

    # Now you can return data_new or pass it to your insert_data_analyst function
    return data_new, sen_status, span_status

def ichimoku_comparation(stock):
    print("ichimoku_project function called")
    
    # Define constants
    DAYS_9 = 9
    DAYS_26 = 26
    DAYS_52 = 52

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)

    # Fetch data and handle potential errors
    try:
        data = pdr.get_data_yahoo(stock, start=start_date, end=end_date.strftime("%Y-%m-%d"))
        data = data.iloc[:int(len(data)*0.8)]
        print("\n\nFetched data for", stock)
    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return None
    
    # Calculate 'n' as the length of the DataFrame
    n = len(data)
    
    if data is None:
        print("Data is None after fetching")
    else:
        print("Data fetched successfully")

    high_9_days = data.High.rolling(DAYS_9).max() 
    low_9_days = data.High.rolling(DAYS_9).min()   
    high_26_days = data.High.rolling(DAYS_26).max()
    low_26_days = data.High.rolling(DAYS_26).min()
    high_52_days = data.High.rolling(DAYS_52).max()
    low_52_days = data.High.rolling(DAYS_52).min()
    
    data['tenkan_sen'] = (high_9_days + low_9_days) / 2
    data['kijun_sen'] = (high_26_days + low_26_days) / 2
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(DAYS_26) 
    data['senkou_span_b'] = ((high_52_days + low_52_days) / 2).shift(DAYS_26)
    data['chikou'] = data.Close.shift(-DAYS_26)
    data = data.iloc[DAYS_26:]
    print(data.dtypes)
    data = data.dropna()
    is_finite = data.applymap(np.isfinite)
    print(is_finite)
    has_na = data.isna().any()
    print(has_na)
    
    # Get the last Kumo cloud components
    last_senkou_span_a = data['senkou_span_a'].iloc[-1]
    last_senkou_span_b = data['senkou_span_b'].iloc[-1]

    # Get the last Kijun-Sen
    last_kijun_sen = data['kijun_sen'].iloc[-1]

    ####################################### TEKAN_SEN FACTOR
    tenkan_sen = data['tenkan_sen']
    trends = get_trend(tenkan_sen)
    tenkan_sen_status = get_tenkan_sen_status(trends)
        
    ######################################## KIJUN_SEN FACTOR
    # Get the last closing price and the last Kijun-Sen value
    last_close = data['Close'].iloc[-1]
    last_kijun_sen = data['kijun_sen'].iloc[-1]

    # Determine the trend based on the position of the closing price relative to the Kijun-Sen line
    if last_close > last_kijun_sen:
        kijun_sen_status = "The market is in an upward trend."
    elif last_close < last_kijun_sen:
        kijun_sen_status = "The market is in a downward trend."
    else:
        kijun_sen_status = "The market is moving sideways."
    print(kijun_sen_status)
        
    ######################################## SENKOU_SEN (KUMO) FACTOR
    # Get the last closing price and the last Senkou Span A and B values
    last_close = data['Close'].iloc[-1]
    last_senkou_span_a = data['senkou_span_a'].iloc[-1]
    last_senkou_span_b = data['senkou_span_b'].iloc[-1]

    # Determine the market trend and potential price movements based on the position of the closing price relative to the Senkou Span A and B lines
    senkou_span_status = get_senkou_span_status(last_close, last_senkou_span_a, last_senkou_span_b)
    print(senkou_span_status)
    
    # Reset the index to move 'Date' from index to columns
    data_reset = data.reset_index()

    # Create a new DataFrame with only the desired columns
    data_new = data_reset[['Date', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']]

    # Now you can return data_new or pass it to your insert_data_analyst function
    return data_new, tenkan_sen_status, kijun_sen_status, senkou_span_status

import pandas as pd

def pembuktian_ichimoku(ichimoku_data):
    array_tren_1_hari = []
    array_tren_1_minggu = []
    array_tren_1_bulan = []

    # Ensure ichimoku_data is a pandas Series
    if isinstance(ichimoku_data, pd.DataFrame):
        ichimoku_data = ichimoku_data['close']
    
    # Calculate Next 1-Day Trend
    for i in range(1, len(ichimoku_data)):  # Start from the second row
        if ichimoku_data.iloc[i] > ichimoku_data.iloc[i - 1]:
            tren_1_hari = 1
        elif ichimoku_data.iloc[i] < ichimoku_data.iloc[i - 1]:
            tren_1_hari = -1
        else:
            tren_1_hari = 0
        array_tren_1_hari.append(tren_1_hari)

    # Calculate Next 1-Week Trend
    for i in range(len(array_tren_1_hari)):
        if i + 7 > len(array_tren_1_hari):
            break  # Stop appending if there's not enough data for a full week
        tren_1_minggu = sum(array_tren_1_hari[i:i + 7])
        array_tren_1_minggu.append(tren_1_minggu)

    # Calculate Next 1-Month Trend
    for i in range(len(array_tren_1_hari)):
        if i + 30 > len(array_tren_1_hari):
            break  # Stop appending if there's not enough data for a full month
        tren_1_bulan = sum(array_tren_1_hari[i:i + 30])
        array_tren_1_bulan.append(tren_1_bulan)

    return array_tren_1_hari, array_tren_1_minggu, array_tren_1_bulan

import pandas as pd
from sqlalchemy import create_engine, text

def get_trend_arrays(stock):
    id_stock = get_emiten_id(stock)
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
    with engine.connect() as conn:
        while True:
            # Fetch data for current date
            query = text("""
                SELECT tenkan_sen, kijun_sen FROM tb_ichimoku_cloud WHERE date = :date AND id_emiten = :id_stock
            """)
            ichimoku_data = conn.execute(query, {'date': current_date, 'id_stock': id_stock}).fetchone()
            
            query = text("""
                SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
            """)
            close_price = conn.execute(query, {'date': current_date, 'id_stock': id_stock}).scalar()
            
            if not ichimoku_data or close_price is None:
                break  # No more data
            
            tenkan_sen, kijun_sen = ichimoku_data
            
            # Calculate status_ichimoku
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

            # Fetch next day close price
            next_day = current_date + pd.Timedelta(days=1)
            next_day_close = conn.execute(text("""
                SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
            """), {'date': next_day, 'id_stock': id_stock}).scalar()
            
            if next_day_close is not None:
                compare_1_hari = 1 if close_price < next_day_close else 0 if close_price > next_day_close else 0.5
                array_1_hari.append(1 if compare_1_hari == status_ichimoku else 0)
            
            # Fetch one week close prices
            week_close_prices = []
            for i in range(1, 8):
                date = current_date + pd.Timedelta(days=i)
                price = conn.execute(text("""
                    SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
                """), {'date': date, 'id_stock': id_stock}).scalar()
                if price is None:
                    break
                week_close_prices.append(price)
            
            if len(week_close_prices) == 7:
                highest_week = max(week_close_prices)
                lowest_week = min(week_close_prices)
                if status_ichimoku == 1:
                    array_1_minggu.append(1 if highest_week > close_price else 0)
                elif status_ichimoku == 0:
                    array_1_minggu.append(1 if lowest_week < close_price else 0)
                else:
                    array_1_minggu.append(1 if highest_week <= close_price or lowest_week >= close_price else 0)
            
            # Fetch one month close prices
            month_close_prices = []
            for i in range(1, 31):
                date = current_date + pd.Timedelta(days=i)
                price = conn.execute(text("""
                    SELECT close FROM tb_detail_emiten WHERE date = :date AND id_emiten = :id_stock
                """), {'date': date, 'id_stock': id_stock}).scalar()
                if price is None:
                    break
                month_close_prices.append(price)
            
            if len(month_close_prices) == 30:
                highest_month = max(month_close_prices)
                lowest_month = min(month_close_prices)
                if status_ichimoku == 1:
                    array_1_bulan.append(1 if highest_month > close_price else 0)
                elif status_ichimoku == 0:
                    array_1_bulan.append(1 if lowest_month < close_price else 0)
                else:
                    array_1_bulan.append(1 if highest_month <= close_price or lowest_month >= close_price else 0)
            
            # Move to the next date
            current_date += pd.Timedelta(days=1)
    
    return array_1_hari, array_1_minggu, array_1_bulan
