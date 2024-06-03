from pandas_datareader import data as pdr
import yfinance as yf
import time
import datetime

import matplotlib.pyplot as plt
import pandas_datareader.data as wb 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

yf.pdr_override()

issuer_stock_codes = ['ADRO.JK', 'AKRA.JK', 'BBRM.JK', 'BYAN.JK', 'DEWA.JK', 'DOID.JK', 'DSSA.JK', 'ELSA.JK', 'GEMS.JK', 'GTSI.JK', 'HITS.JK', 'HRUM.JK', 'INDY.JK', 'ITMG.JK', 'JSKY.JK', 'KKGI.JK', 'LEAD.JK', 'MBSS.JK', 'MCOL.JK', 'MEDC.JK', 'MYOH.JK', 'PGAS.JK', 'PSSI.JK', 'PTBA.JK', 'PTIS.JK', 'PTRO.JK', 'RAJA.JK', 'RMKE.JK', 'SHIP.JK', 'SOCI.JK', 'TEBE.JK', 'TOBA.JK', 'UNIQ.JK', 'WINS.JK', 'ADMR.JK', 'AIMS.JK', 'APEX.JK', 'ARII.JK', 'ARTI.JK', 'BESS.JK', 'BIPI.JK', 'BOSS.JK', 'BSML.JK', 'BSSR.JK', 'BULL.JK', 'BUMI.JK', 'CANI.JK', 'CNKO.JK', 'DWGL.JK', 'ENRG.JK', 'ETWA.JK', 'FIRE.JK', 'GTBO.JK', 'IATA.JK', 'INPS.JK', 'ITMA.JK', 'KOPI.JK', 'MBAP.JK', 'MITI.JK', 'MTFN.JK', 'PKPK.JK', 'RIGS.JK', 'RUIS.JK']

for assets in issuer_stock_codes:
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)
    data = pdr.get_data_yahoo(assets, start=start_date, end=end_date.strftime("%Y-%m-%d"))
    df = data
    
    # Create a new dataframe with only the 'Close' column 
    dataf = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = dataf.values
    # Get the number of rows to train the model on
    training_data_len = max(60, int(np.ceil(len(dataset) * .100)))
    
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # x_train.shape

    from keras.models import Sequential
    from keras.layers import Dense, LSTM

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Find the min and max values of 'Close' column in the first DataFrame
    # y_min = df['Close'].min()
    # y_max = df['Close'].max()

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
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

    plt.savefig(f'prediction_{assets}.png')
    plt.clf()
    
    