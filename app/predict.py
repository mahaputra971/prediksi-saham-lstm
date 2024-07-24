# app/predict.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from app.sql import get_emiten_id, get_model_id_by_emiten, fetch_stock_data, load_model_from_db, get_table_data
import tempfile
from datetime import datetime, timedelta
from integrations.ic_project import ichimoku_sql, interpret_sen_status, get_senkou_span_status

from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
import pandas as pd

def predict_with_loaded_model(stock, start_date, end_date):
    # Get emiten ID
    stock_id = get_emiten_id(stock)
    if stock_id is None:
        print(f"Stock ID for {stock} not found.")
        return None, None

    # Get model ID dynamically
    model_id = get_model_id_by_emiten(stock_id)
    if model_id is None:
        print(f"Model ID for emiten {stock_id} not found.")
        return None, None

    # Fetch stock data
    data = fetch_stock_data([stock], start_date, end_date)
    if data is None or stock not in data:
        print(f"No data found for stock {stock} from {start_date} to {end_date}.")
        return None, None

    company_df = data[stock]
    if company_df.empty:
        print(f"No data available for stock {stock} in the given date range.")
        return None, None

    # Load the model from the database
    model = load_model_from_db(model_id)
    if model is None:
        print(f"Model with ID {model_id} could not be loaded.")
        return None, None

    # Prepare the data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(company_df['Close'].values.reshape(-1, 1))

    # Create the test dataset
    if len(company_df) < 60:
        print(f"Not enough data to make predictions for {stock} from {start_date} to {end_date}.")
        return None, None

    test_data = scaled_data[-(60 + len(company_df)):]

    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    if x_test.shape[0] == 0 or x_test.shape[1] == 0:
        print(f"Not enough data points after preprocessing for {stock} from {start_date} to {end_date}.")
        return None, None
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
    plot_path = f"/static/predictions/{stock}_{start_date}_to_{end_date}.png"
    plt.savefig(f"app{plot_path}")
    plt.close()

    # Print evaluation metrics
    print(f'\nMean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape}%')
    
    # Find the highest and lowest prices
    highest_price = np.max(predictions)
    lowest_price = np.min(predictions)
    # Find the dates when the highest and lowest prices occur
    highest_price_date = company_df.index[-len(predictions) + np.argmax(predictions)]
    lowest_price_date = company_df.index[-len(predictions) + np.argmin(predictions)]
    # Print the highest and lowest prices and their corresponding dates
    print(f'Highest Price: {highest_price} IDR on {highest_price_date}')
    print(f'Lowest Price: {lowest_price} IDR on {lowest_price_date}')

    # Return accuracy metrics and plot path
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "Highest Price": highest_price,
        "Lowest Price": lowest_price,
        "Highest Price Date": highest_price_date,
        "Lowest Price Date": lowest_price_date
    }, plot_path

def predict_future(stock, future_days):
    data = yf.download(stock)
    data = data.filter(['Close'])
    dataset = data.values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    data_for_prediction = scaled_data[-(future_days + 60):]
    x_future = []

    for i in range(60, len(data_for_prediction)):
        x_future.append(data_for_prediction[i-60:i, 0])

    x_future = np.array(x_future)
    x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
    
    stock_id = get_emiten_id(stock)
    if stock_id is None:
        print(f"Stock ID for {stock} not found.")
        return None, None

    # Get model ID dynamically
    model_id = get_model_id_by_emiten(stock_id)
    if model_id is None:
        print(f"Model ID for emiten {stock_id} not found.")
        return None, None 
    
    model = load_model_from_db(model_id)
    if model is None:
        print(f"Model with ID {model_id} could not be loaded.")
        return None, None

    future_predictions = model.predict(x_future)
    future_predictions = scaler.inverse_transform(future_predictions)

    future_dates = pd.date_range(datetime.now() + timedelta(days=1), periods=future_days, freq='D')

    plt.figure(figsize=(16, 6))
    plt.title(f'Predicted Close Price for the Next {future_days} Days')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price IDR', fontsize=18)
    plt.plot(future_dates, future_predictions, color='r', label='Predicted Price')
    plt.legend()
    plot_path = f"/static/predictions/{stock}future.png"
    plt.savefig(f"app{plot_path}")
    plt.close()

    print(future_predictions)

    highest_prediction = future_predictions.max()
    lowest_prediction = future_predictions.min()

    max_price_date = future_dates[future_predictions.argmax()]
    min_price_date = future_dates[future_predictions.argmin()]

    print(f'Prediction Harga tertinggi: {highest_prediction} pada tanggal {max_price_date.strftime("%Y-%m-%d")}')
    print(f'Prediction Harga terendah: {lowest_prediction} pada tanggal {min_price_date.strftime("%Y-%m-%d")}')

    predictions = (highest_prediction, lowest_prediction, max_price_date, min_price_date)

    return predictions, plot_path

def ichimoku_predict(stock, specific_date): 
    # Mengambil data Ichimoku dari database
    data_ichimoku = get_table_data(stock, 'tb_data_ichimoku_cloud')
    # print(f'186: {data_ichimoku}\n')
    # print(f'187: {specific_date}')

    # Konversi daftar dictionary menjadi DataFrame
    df = pd.DataFrame(data_ichimoku)
    print(f'191: {df}')

    # Konversi kolom 'date' menjadi datetime
    df['date'] = pd.to_datetime(df['date'])
    # print(f'195: {df["date"]}')
    # print(f'195 (dtype): {df["date"].dtype}')

    # Konversi specific_date menjadi datetime jika dalam format string
    if isinstance(specific_date, str):
        specific_date = datetime.strptime(specific_date, '%Y-%m-%d')
    elif isinstance(specific_date, datetime):
        specific_date = specific_date.date()  # Pastikan specific_date dalam tipe date
    
    # Konversi specific_date menjadi dtype yang sama dengan kolom df['date']
    specific_date = pd.to_datetime(specific_date)
    # print(f'201: specific_date as datetime: {specific_date}')
    # print(f'201 (dtype): {type(specific_date)}')

    # Memilih baris dengan tanggal yang spesifik
    row = df.loc[df['date'] == specific_date]
    # print(f'205: row before while loop: {row}')

    # Jika baris 'row' kosong, decrement tanggal satu hari dan coba lagi
    while row.empty:
        specific_date -= timedelta(days=1)
        # print(f'209: specific_date in while loop: {specific_date}')
        if specific_date.year < 1970:
            raise ValueError("No data available for the specified date and previous dates.")
        row = df.loc[df['date'] == specific_date]
        # print(f'213: row in while loop: {row}')

    # Sekarang 'row' berisi data untuk tanggal pertama yang memiliki data
    tenkan_sen = row['tenkan_sen'].values[0]
    kijun_sen = row['kijun_sen'].values[0]
    senkou_span_a = row['senkou_span_a'].values[0]
    senkou_span_b = row['senkou_span_b'].values[0]
    print(f'228: {tenkan_sen}')
    print(f'229: {kijun_sen}')
    print(f'230: {senkou_span_a}')
    print(f'231: {senkou_span_b}')
    
    # Mendownload data harga saham
    data = yf.download(stock)
    close_value = data.loc[specific_date.strftime('%Y-%m-%d'), 'Close']
    print(f'236: {close_value}')
    
    # Mendapatkan status Senkou Span dan status Sen
    span_status = get_senkou_span_status(close_value, senkou_span_a, senkou_span_b)
    sen_status = interpret_sen_status(close_value, tenkan_sen, kijun_sen) 
    print(f'241: {span_status}')
    print(f'242: {sen_status}')
    
    return span_status, sen_status