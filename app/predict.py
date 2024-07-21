# app/predict.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from app.sql import get_emiten_id, get_model_id_by_emiten, fetch_stock_data, load_model_from_db
import tempfile

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

    # Return accuracy metrics and plot path
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape
    }, plot_path
