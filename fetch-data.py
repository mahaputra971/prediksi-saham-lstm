import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
tf.device('/GPU:0')
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

from datetime import datetime, timedelta

import importlib
import sql

importlib.reload(sql)
from sql import show_tables, insert_tables, show_specific_tables, get_issuer

row = get_issuer()
# for r in range(j):
for stock in row:
    stock_data = [stock]
    print()
    print(stock_data)

    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)

    data = {}

    for stock in stock_data:
        data[stock] = yf.download(stock, start, end)
        while data[stock].empty and start < end:
            start += timedelta(days=1)
            data[stock] = yf.download(stock, start, end)

    company_list = [data[stock] for stock in stock_data]
    company_name = [stock]

    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name

    df = pd.concat(company_list, axis=0)

    # Summary Stats and General Info
    print(df.describe())
    print(df.info())

    # Historical closing price
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(1, 1, i)
        company['Adj Close'].plot()
        plt.ylabel('Adj Close')
        plt.xlabel(None)
        plt.title(f"Closing Price of {company_name[i - 1]}")
        
    plt.tight_layout()
    plt.savefig(f'picture/closing_price/{stock}.png')
    # plt.show()  # Ensure plots are displayed

    # Total volume of stock traded each day
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(top=1.25, bottom=1.2)

    for i, company in enumerate(company_list, 1):
        plt.subplot(1, 1, i)
        company['Volume'].plot()
        plt.ylabel('Volume')
        plt.xlabel(None)
        plt.title(f"Sales Volume for {company_name[i - 1]}")

    plt.tight_layout()
    plt.savefig(f'picture/sales_volume/{stock}.png')
    # plt.show()  # Ensure plots are displayed
    
    # end = datetime.now()
    # start = datetime(end.year-10, end.month, end.day)
    
    # # Get the stock quote
    # df = pdr.get_data_yahoo(stock, start=start, end=end)

    plt.figure(figsize=(16, 6))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.savefig(f'picture/close_price_history/{stock}.png')
    # plt.show()

    data = df.filter(['Close'])
    dataset = data.values

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    split_index = int(len(scaled_data) * 0.8)
    # Split data into training and testing sets
    print("106, this is the split_index fetch-data.py: ", split_index)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]

    print("109, this is the train_data fetch-data.py: ", train_data)
    print("109, this is the train_data.length fetch-data.py: ", len(train_data))
    print("110, this is the test_data fetch-data.py: ", test_data)

    x_train = []
    y_train = []

    if len(train_data) >= 60:
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        # print('122 x_train', x_train)
        # print('123 y_train: ', y_train)
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    else:
        print("Not enough data in train_data")
    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    ##################################################################################
    x_test = []
    y_test = dataset[split_index:, :]
    if len(test_data) >= 60:
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            y_test.append(test_data[i, 0])
        # print('122 x_train', x_train)
        # print('123 y_train: ', y_train)
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    else:
        print("Not enough data in train_data")

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
    rmse
    
    # Plot the data
    train = data[:split_index]
    valid = data[split_index:]
    valid['Predictions'] = predictions

    # Visualize the data
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')

    # Set the limits of the Y-axis to match the first plot
    # plt.ylim(y_min, y_max)

    plt.show()