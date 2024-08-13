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
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

import importlib
import integrations
importlib.reload(integrations)
from integrations import ichimoku_project, ichimoku_sql, pembuktian_ichimoku, get_issuer, get_emiten_id, insert_data_analyst, save_model_to_db, load_model_from_db, get_model_id_by_emiten, save_model_to_directory, load_model_from_directory

# Setup the SQLAlchemy engine and session
# engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')
# Session = sessionmaker(bind=engine)
# session = Session()

from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables

# Create the engine
engine = create_engine(os.getenv('MYSQL_STRING'))
Session = sessionmaker(bind=engine)
session = Session()
today = datetime.now().strftime("%Y-%m-%d")

# Stock data
# stock_data = ['TLKM.JK', 'BBRI.JK', 'ASII.JK', 'BMRI.JK', 'KLBF.JK', 'UNVR.JK', 'MTDL.JK', 'INDF.JK', 'ACES.JK']
# company_name = stock_data 
# stock_nama_data = company_name

def fetch_stock_data(stock_list, start, end):
    data = {stock: yf.download(stock, start, end) for stock in stock_list}
    return data

def plot_stock_data(company, column, xlabel, ylabel, title, folder_name, stock):
    plt.figure(figsize=(16, 9))
    company[column].plot()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(f'picture/{folder_name}/{stock}.png')
    plt.close()

def train_and_evaluate_model(df, stock_name):
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
    valid.loc[:, 'Predictions'] = predictions

    plt.figure(figsize=(16, 6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    # f'picture/{folder_name}/{stock}.png'
    plt.savefig(f'picture/accuracy/{stock_name}.png')
    plt.close()

    print(valid)

    mae = mean_absolute_error(valid['Close'], valid['Predictions'])
    print(f"Mean Absolute Error (MAE): {mae}")

    mse = mean_squared_error(valid['Close'], valid['Predictions'])
    print(f"Mean Squared Error (MSE): {mse}")

    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    mape = mean_absolute_percentage_error(valid['Close'], valid['Predictions'])
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

    highest_prediction = valid['Close'].max()
    lowest_prediction = valid['Close'].min()

    highest_date = valid['Close'].idxmax()
    lowest_date = valid['Close'].idxmin()

    print(f"Highest prediction: {highest_prediction} on {highest_date}")
    print(f"Lowest prediction: {lowest_prediction} on {lowest_date}")

    return model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape, valid

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def predict_future(model, scaler, scaled_data, future_days, stock):
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
    # plt.savefig(f'picture/prediction/{future_dates[0].strftime("%Y-%m-%d")}_to_{future_dates[-1].strftime("%Y-%m-%d")}_future_predictions.png')
    plt.savefig(f'picture/prediction/{stock}.png')
    plt.close()

    print(future_predictions)

    highest_prediction = future_predictions.max()
    lowest_prediction = future_predictions.min()

    max_price_date = future_dates[future_predictions.argmax()]
    min_price_date = future_dates[future_predictions.argmin()]

    print(f'Prediction Harga tertinggi: {highest_prediction} pada tanggal {max_price_date.strftime("%Y-%m-%d")}')
    print(f'Prediction Harga terendah: {lowest_prediction} pada tanggal {min_price_date.strftime("%Y-%m-%d")}')

    return highest_prediction, lowest_prediction, max_price_date, min_price_date

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 100, end.month, end.day)

# Process each stock separately
def engine_main(stock):
    print(f"Processing stock: {stock}")
    
    # Fetch stock data
    data = yf.download(stock)
    
    if data.empty:
        print(f"Stock {stock} not found in data.")
        return
    
    company_df = data
    
    # Summary Stats and General Info
    print(company_df.describe())
    print(company_df.info())

    # Plotting historical adjusted closing price
    plot_stock_data(company_df, 'Adj Close', 'Adj Close', None, f'Closing Price of {stock}', 'adj_closing_price', stock)

    # Plotting sales volume
    plot_stock_data(company_df, 'Volume', 'Volume', None, f'Sales Volume of {stock}', 'sales_volume', stock)

    # Getting historical data for the past 100 years
    historical_start = datetime.now() - relativedelta(years=100)
    historical_data = fetch_stock_data([stock], historical_start, datetime.now())
    historical_df = historical_data[stock]
    print(historical_df.tail())

    # Plotting historical closing price
    plot_stock_data(historical_df, 'Close', 'Date', f'Close Price IDR {stock}', 'Close Price History', 'close_price_history', stock)

    # Training and evaluating the model
    model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape, valid = train_and_evaluate_model(historical_df, stock)

    # Menyimpan model ke database
    stock_id = get_emiten_id(stock)
    model_name = f'LSTM Model for {stock}'
    algorithm = 'LSTM'
    hyperparameters = model.get_config()
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape
    }
    # save_model_to_db(model, stock_id, model_name, algorithm, hyperparameters, metrics)
    save_model_to_directory(model, stock_id, model_name, algorithm, hyperparameters, metrics)

    # Setting up for future predictions
    future_prediction_period = int(len(scaled_data) * 0.1)

    # Predicting future prices
    max_price, min_price, max_price_date, min_price_date = predict_future(model, scaler, scaled_data, future_prediction_period, stock)

    # BUAT LOGIC UNTUK TAMBAHIN KE DATABASE

    # id for fk in insert
    stock_id = get_emiten_id(stock)

    # Save data to table 'tb_detail_emiten'
    df_copy = historical_df.reset_index()
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

    data_lstm = {
        'id_emiten': stock_id,
        'RMSE': rmse,
        'MAPE': mape,
        'MAE': mae,
        'MSE': mse,
        'date': datetime.now().strftime("%Y-%m-%d")
    }
    insert_data_analyst("tb_lstm", data_lstm)

    # Call the ichimoku_project function
    data_ic, sen_status, span_status = ichimoku_sql(stock)
    data_ic = pd.DataFrame(data_ic)
    data_ic['id_emiten'] = stock_id
    insert_data_analyst('tb_data_ichimoku_cloud', data_ic)

    data_ic_status = {
        'id_emiten': stock_id,
        'sen_status': sen_status,
        'span_status': span_status,
        'date': datetime.now().strftime("%Y-%m-%d")
    }
    insert_data_analyst('tb_ichimoku_status', data_ic_status)

    # Save data to table 'tb_prediction_lstm'
    data_prediction_lstm = {
        'id_emiten': stock_id,
        'max_price': max_price,
        'min_price': min_price,
        'max_price_date': max_price_date.strftime("%Y-%m-%d"),
        'min_price_date': min_price_date.strftime("%Y-%m-%d"),
        'date': datetime.now().strftime("%Y-%m-%d")
    }
    insert_data_analyst('tb_prediction_lstm', data_prediction_lstm)

    # Save data to table 'tb_lstm_summary'
    #f'picture/{folder_name}/{stock}.png'
    date_save = datetime.now().strftime("%Y-%m-%d")
    img_closing_price = Image.open(f'picture/adj_closing_price/{stock}.png')
    img_sales_volume = Image.open(f'picture/sales_volume/{stock}.png')
    img_price_history = Image.open(f'picture/close_price_history/{stock}.png')
    img_comparation = Image.open(f'picture/accuracy/{stock}.png')
    img_prediction = Image.open(f'picture/prediction/{stock}.png')
    img_ichimoku_cloud = Image.open(f'picture/ichimoku/{stock}.png')
    data_summary = {
        'id_emiten': stock_id,
        'pic_closing_price': img_closing_price,
        'pic_sales_volume': img_sales_volume, 
        'pic_price_history': img_price_history,
        'pic_comparation': img_comparation,
        'pic_prediction': img_prediction,
        'pic_ichimoku_cloud': img_ichimoku_cloud,
        'render_date': date_save
    }
    insert_data_analyst('tb_summary', data_summary)

    # Save data to table 'tb_accuracy_ichimoku_cloud'
    tren_1hari_sen, tren_1minggu_sen, tren_1bulan_sen = pembuktian_ichimoku(stock, 'sen')
    tren_1hari_span, tren_1minggu_span, tren_1bulan_span = pembuktian_ichimoku(stock, 'span')

    percent_1_hari_sen = pd.Series(tren_1hari_sen).mean() * 100
    percent_1_minggu_sen = pd.Series(tren_1minggu_sen).mean() * 100
    percent_1_bulan_sen = pd.Series(tren_1bulan_sen).mean() * 100

    percent_1_hari_span = pd.Series(tren_1hari_span).mean() * 100
    percent_1_minggu_span = pd.Series(tren_1minggu_span).mean() * 100
    percent_1_bulan_span = pd.Series(tren_1bulan_span).mean() * 100

    print(f"Accuracy tren 1 hari SEN: {percent_1_hari_sen}%")
    print(f"Accuracy tren 1 minggu SEN: {percent_1_minggu_sen}%")
    print(f"Accuracy tren 1 bulan SEN: {percent_1_bulan_sen}%")

    print(f"\nAccuracy tren 1 hari SPAN: {percent_1_hari_span}%")
    print(f"Accuracy tren 1 minggu SPAN: {percent_1_minggu_span}%")
    print(f"Accuracy tren 1 bulan SPAN: {percent_1_bulan_span}%")
    data_accuracy_ichimoku = {
        'id_emiten': stock_id,
        'percent_1_hari_sen': percent_1_hari_sen,
        'percent_1_minggu_sen': percent_1_minggu_sen,
        'percent_1_bulan_sen': percent_1_bulan_sen,
        'percent_1_hari_span': percent_1_hari_span,
        'percent_1_minggu_span': percent_1_minggu_span,
        'percent_1_bulan_span': percent_1_bulan_span,
        'date': date_save
    }
    insert_data_analyst('tb_accuracy_ichimoku_cloud', data_accuracy_ichimoku)
    
    # Set the 'status' column in 'tb_emiten' to '1' for the given stock
    try:
        update_query = text("UPDATE tb_emiten SET status = :status WHERE kode_emiten = :stock")
        session.execute(update_query, {'status': 1, 'stock': stock})
        session.commit()
        print("Commit success")
    except Exception as e:
        print(f"Commit error: {str(e)}")

# Fungsi untuk memprediksi dengan model yang dimuat
def predict_with_loaded_model(stock, start_date, end_date):
    # Get emiten ID
    stock_id = get_emiten_id(stock)
    if stock_id is None:
        print(f"Stock ID for {stock} not found.")
        return

    # Get model ID dynamically
    # model_id = get_model_id_by_emiten(stock_id)
    # if model_id is None:
    #     print(f"Model ID for emiten {stock_id} not found.")
    #     return

    # Fetch stock data
    data = fetch_stock_data([stock], start_date, end_date)
    company_df = data[stock]

    # Load the model from the database
    # model = load_model_from_db(model_id)
    model_name = f'LSTM Model for {stock}'
    model = load_model_from_directory(model_name)
    if model is None:
        print(f"Model with ID {stock} could not be loaded.")
        return

    # Prepare the data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(company_df['Close'].values.reshape(-1, 1))

    # Create the test dataset
    if len(company_df) < 60:
        print(f"Not enough data to make predictions for {stock} from {start_date} to {end_date}.")
        return

    test_data = scaled_data[-(60 + len(company_df)):]

    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    if x_test.shape[0] == 0 or x_test.shape[1] == 0:
        print(f"Not enough data points after preprocessing for {stock} from {start_date} to {end_date}.")
        return
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Calculate evaluation metrics
    actual = company_df['Close'].values[-len(predictions):]
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predictions)

    # Plot the predictions
    plt.figure(figsize=(16, 6))
    plt.title(f'Predicted Close Price for {stock} from {start_date} to {end_date}')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.plot(company_df.index[-len(predictions):], predictions, color='r', label='Predicted Price')
    plt.plot(company_df.index[-len(predictions):], actual, color='b', label='Actual Price')
    plt.legend()
    # plt.savefig(f'picture/prediction/{stock}_{start_date}_to_{end_date}_predictions_{today}.png')
    plt.savefig(f'picture/prediction/{stock}.png')
    plt.close()

    # Print predictions
    print(f'Predicted prices for {stock} from {start_date} to {end_date}:')
    for date, pred_price in zip(company_df.index[-len(predictions):], predictions):
        print(f'{date}: {pred_price[0]}')

    # Print evaluation metrics
    print(f'\nMean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

    # Finding highest and lowest prices and their dates
    highest_price = company_df['Close'].max()
    lowest_price = company_df['Close'].min()
    highest_date = company_df['Close'].idxmax()
    lowest_date = company_df['Close'].idxmin()

    print(f'\nHighest actual price: {highest_price} on {highest_date}')
    print(f'Lowest actual price: {lowest_price} on {lowest_date}')
    


# Usage example
# stock = 'BELI.JK'  # Replace with the stock ticker
# start_date = '2023-03-01'  # Replace with the start date for the prediction
# end_date = '2023-07-31'  # Replace with the end date for the prediction

# predict_with_loaded_model(stock, start_date, end_date)
