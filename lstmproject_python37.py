import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()

from datetime import timedelta, datetime
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker 

import importlib
import sql
import ic_project

importlib.reload(ic_project)
importlib.reload(sql)

from sql import show_tables, insert_tables, show_specific_tables, get_issuer, insert_data_analyst, get_emiten_id
from ic_project import ichimoku_project, ichimoku_sql

engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')
Session = sessionmaker(bind=engine)
session = Session()

stock_data = ['BELI.JK']
company_name = stock_data 
stock_nama_data = company_name

def fetch_stock_data(stock_list, start, end):
    data = {stock: yf.download(stock, start, end) for stock in stock_list}
    return data

def plot_stock_data(stock, company, column, xlabel, ylabel, title, folder_name):
    plt.figure(figsize=(16, 9))
    company[column].plot()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f'picture/{folder_name}/{stock}.png')
    # plt.show()

def train_and_evaluate_model(df):
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60:]
    x_test, y_test = [], dataset[training_data_len:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig(f'picture/accuracy/{stock}.png')
    # plt.show()

    print(valid)

    mae = mean_absolute_error(valid['Close'], valid['Predictions'])
    print(f"Mean Absolute Error (MAE): {mae}")

    mse = mean_squared_error(valid['Close'], valid['Predictions'])
    print(f"Mean Squared Error (MSE): {mse}")

    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    mape = mean_absolute_percentage_error(valid['Close'], valid['Predictions'])
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

    return model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict_future(model, scaler, scaled_data, future_days):
    data_for_prediction = scaled_data[-(future_days + 60):]
    x_future = []

    for i in range(60, len(data_for_prediction)):
        x_future.append(data_for_prediction[i-60:i, 0])

    x_future = np.array(x_future)
    x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

    future_predictions = model.predict(x_future)
    future_predictions = scaler.inverse_transform(future_predictions)

    future_dates = pd.date_range(datetime.now() + timedelta(days=1), periods=future_days, freq='D')

    plt.figure(figsize=(16, 6))
    plt.title(f'Predicted Close Price for the Next {future_days} Days')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.plot(future_dates, future_predictions)
    plt.savefig('future_predictions.png')
    plt.savefig(f'picture/prediction/{stock}.png')
    # plt.show()

    print(future_predictions)

    max_price = future_predictions.max()
    min_price = future_predictions.min()
    max_price_date = future_dates[future_predictions.argmax()]
    min_price_date = future_dates[future_predictions.argmin()]

    print(f'Harga tertinggi: {max_price} pada tanggal {max_price_date.strftime("%Y-%m-%d")}')
    print(f'Harga terendah: {min_price} pada tanggal {min_price_date.strftime("%Y-%m-%d")}')

    return max_price, min_price, max_price_date, min_price_date
    
# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 100, end.month, end.day)

# Process each stock separately
for stock, stock_nama in zip(stock_data, stock_nama_data):
    print(f"Processing stock: {stock}")

    data = fetch_stock_data([stock], start, end)
    company_df = data[stock]

    plot_stock_data(stock, company_df, 'Adj Close', 'Tanggal', 'Harga Penutupan Disesuaikan (IDR)', f"Harga Penutupan Disesuaikan {stock}", 'adj_closing_price')
    plot_stock_data(stock, company_df, 'Volume', 'Tanggal', 'Volume Saham', f"Volume Saham {stock}", 'sales_volume')
    plot_stock_data(stock, company_df, 'Close', 'Tanggal', 'Harga Penutupan (IDR)', f"Riwayat Harga Penutupan {stock}", 'close_price_history')

    model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape = train_and_evaluate_model(company_df)
    
    insert_tables(stock, stock_nama, mae, mse, rmse, mape, 'LSTM')

    max_price, min_price, max_price_date, min_price_date = predict_future(model, scaler, scaled_data, future_days=30)
    
    emiten_id = get_emiten_id(stock)
    
    insert_data_analyst(stock, 'LSTM', 'Adj Close', mae, mse, rmse, mape, max_price, min_price, max_price_date.strftime('%Y-%m-%d'), min_price_date.strftime('%Y-%m-%d'), emiten_id)
    
    ichimoku_sql(stock)

    # plt.show()
