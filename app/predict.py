# app/predict.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from app.sql import get_emiten_id, get_model_id_by_emiten, fetch_stock_data, load_model_from_db
import tempfile
from datetime import datetime, timedelta

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