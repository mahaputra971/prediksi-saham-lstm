from pandas_datareader import data as pdr
import yfinance as yf
import datetime
import json
import mongo
import matplotlib.pyplot as plt
from sql import insert_data_analyst
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
        kijun_sen_status = "The market is in an upward trend."
    elif last_close < last_kijun_sen:
        kijun_sen_status = "The market is in a downward trend."
    else:
        kijun_sen_status = "The market is moving sideways."
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
    plt.savefig(f'ichimokuproject/image/plot_IC_{stock}.png')
    plt.clf()  # Clear the current figure to start a new plot for the next asset
    
    # print(data)

    print ("\nKode Emiten (test) : ", stock)
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