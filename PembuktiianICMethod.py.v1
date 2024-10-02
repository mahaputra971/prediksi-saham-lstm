import datetime
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

yf.pdr_override()

issuer_stock_codes = 'BYAN.JK'
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)
data = pdr.get_data_yahoo(issuer_stock_codes, start=start_date, end=end_date.strftime("%Y-%m-%d"))

high9 = data.High.rolling(9).max()
low9 = data.High.rolling(9).min()
high26 = data.High.rolling(26).max()
low26 = data.High.rolling(26).min()
high52 = data.High.rolling(52).max()
low52 = data.High.rolling(52).min()

data['tenkan_sen'] = (high9 + low9) / 2
data['kijun_sen'] = (high26 + low26) / 2
data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
data['senkou_span_b'] = ((high52 + low52) / 2).shift(26)
data['chikou'] = data.Close.shift(-26)
data = data.iloc[26:]

plt.plot(data.index, data['tenkan_sen'], lw=0.7, color='red')
plt.plot(data.index, data['kijun_sen'], lw=0.7, color='yellow')
plt.plot(data.index, data['chikou'], lw=0.7, color='grey')
plt.plot(data.index, data['Close'], lw=0.7, color='green')
plt.title("Ichimoku Saham : " + str(issuer_stock_codes))
plt.ylabel("Harga")
kumo = data['Adj Close'].plot(lw=0.7, color='blue')
kumo.fill_between(data.index, data.senkou_span_a, data.senkou_span_b, where=data.senkou_span_a >= data.senkou_span_b, color='lightgreen')
kumo.fill_between(data.index, data.senkou_span_a, data.senkou_span_b, where=data.senkou_span_a < data.senkou_span_b, color='lightcoral')
plt.grid()
plt.show()
plt.clf()

print("\nKode Emiten:", issuer_stock_codes)
print("Close Price:", data['Close'].iloc[-1])
print("tenkan_sen:", data['tenkan_sen'].iloc[-1])
print("kijun_sen:", data['kijun_sen'].iloc[-1])
print("senkou_span_a:", data['senkou_span_a'].iloc[-1])
print("senkou_span_b:", data['senkou_span_b'].iloc[-1])

# TEKAN_SEN FACTOR
tenkan_sen = data['tenkan_sen']
x = np.array(range(len(tenkan_sen))).reshape(-1, 1)
y = tenkan_sen.values.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]

if slope > 0:
    print("The Tenkan-Sen is in an uptrend.")
elif slope < 0:
    print("The Tenkan-Sen is in a downtrend.")
else:
    print("The Tenkan-Sen is moving sideways.")

# KIJUN_SEN FACTOR
last_close = data['Close'].iloc[-1]
last_kijun_sen = data['kijun_sen'].iloc[-1]

if last_close > last_kijun_sen:
    print("The market is in an upward trend.")
elif last_close < last_kijun_sen:
    print("The market is in a downward trend.")
else:
    print("The market is moving sideways.")

# SENKOU_SEN (KUMO) FACTOR
last_close = data['Close'].iloc[-1]
last_senkou_span_a = data['senkou_span_a'].iloc[-1]
last_senkou_span_b = data['senkou_span_b'].iloc[-1]

if last_close > last_senkou_span_a and last_senkou_span_a > last_senkou_span_b:
    print("Status: Uptrend")
elif last_close < last_senkou_span_a and last_senkou_span_a < last_senkou_span_b:
    print("Status: Downtrend")
elif last_close < last_senkou_span_b and last_senkou_span_a > last_senkou_span_b:
    print("Status: Will Dump")
elif last_close > last_senkou_span_b and last_senkou_span_a < last_senkou_span_b:
    print("Status: Will Pump")
elif last_senkou_span_b < last_close < last_senkou_span_a and last_senkou_span_a > last_senkou_span_b:
    print("Status: Uptrend and Will Bounce Up")
elif last_senkou_span_b < last_close < last_senkou_span_a and last_senkou_span_a < last_senkou_span_b:
    print("Status: Downtrend and Will Bounce Down")

# TEKAN_SEN
data['tenkan_sen_trend'] = data['tenkan_sen'].diff()
data['tenkan_sen_trend'] = data['tenkan_sen_trend'].apply(lambda x: 'uptrend' if x > 0 else 'downtrend')
data['future_close_trend'] = data['Close'].diff().shift(-1)
data['future_close_trend'] = data['future_close_trend'].apply(lambda x: 'rise' if x > 0 else 'fall')
data = data[:-1]

le = LabelEncoder()
data['tenkan_sen_trend'] = le.fit_transform(data['tenkan_sen_trend'])
data['future_close_trend'] = le.fit_transform(data['future_close_trend'])

features = data['tenkan_sen_trend'].values.reshape(-1, 1)
labels = data['future_close_trend'].values

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(features_train, labels_train)

predictions = model.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions, average='micro')
recall = recall_score(labels_test, predictions, average='micro')
f1 = f1_score(labels_test, predictions, average='micro')

print("\nTENKAN SEN VALUE")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

accuracy_percentage = accuracy * 100
print("Accuracy: {:.2f}%".format(accuracy_percentage))

# KIJUN_SEN
data.loc[:, 'trend'] = np.where(data['Close'] > data['kijun_sen'], 'uptrend', 'downtrend')
data.loc[:, 'trend'] = data['trend'].shift(-1)
data = data[:-1]

features = np.column_stack((data['Close'], data['kijun_sen']))
labels = data['trend']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(features_train, labels_train)

predictions = model.predict(features_test)

accuracy = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions, average='micro')
recall = recall_score(labels_test, predictions, average='micro')
f1 = f1_score(labels_test, predictions, average='micro')

percentage_accuracy = accuracy * 100

print("\nKIJUN SEN VALUE")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print(f"Percentage Accuracy of Predicting Trend: {percentage_accuracy}%")

# SENKOU_SPAN
df = data

features = np.array([df['Close'], df['senkou_span_a'], df['senkou_span_b']]).T
labels = np.array(df['trend'])

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(features_train, labels_train)

predictions = model.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)
precision = precision_score(labels_test, predictions, pos_label='uptrend')
recall = recall_score(labels_test, predictions, pos_label='uptrend')
f1 = f1_score(labels_test, predictions, pos_label='uptrend')

uptrend_condition = (df['Close'] > df['senkou_span_a']) & (df['senkou_span_a'] > df['senkou_span_b'])
downtrend_condition = (df['Close'] < df['senkou_span_a']) & (df['senkou_span_a'] < df['senkou_span_b'])
dump_condition = (df['Close'] < df['senkou_span_b']) & (df['senkou_span_a'] > df['senkou_span_b'])
pump_condition = (df['Close'] > df['senkou_span_b']) & (df['senkou_span_a'] < df['senkou_span_b'])
bouncing_up_condition = ((df['senkou_span_b'] < df['Close']) & (df['Close'] < df['senkou_span_a'])) & (df['senkou_span_a'] > df['senkou_span_b'])
bouncing_down_condition = ((df['senkou_span_b'] > df['Close']) & (df['Close'] > df['senkou_span_a'])) & (df['senkou_span_a'] < df['senkou_span_b'])

uptrend_accuracy = accuracy_score(labels[uptrend_condition], model.predict(features[uptrend_condition])) * 100
downtrend_accuracy = accuracy_score(labels[downtrend_condition], model.predict(features[downtrend_condition])) * 100
dump_accuracy = accuracy_score(labels[dump_condition], model.predict(features[dump_condition])) * 100
pump_accuracy = accuracy_score(labels[pump_condition], model.predict(features[pump_condition])) * 100
bouncing_up_accuracy = accuracy_score(labels[bouncing_up_condition], model.predict(features[bouncing_up_condition])) * 100
bouncing_down_accuracy = accuracy_score(labels[bouncing_down_condition], model.predict(features[bouncing_down_condition])) * 100

print("\nSENKAN SEN")
print("Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Percentage Accuracy for Each Market Condition:")
print(f"Uptrend Accuracy: {uptrend_accuracy}%")
print(f"Downtrend Accuracy: {downtrend_accuracy}%")
print(f"Dump Accuracy: {dump_accuracy}%")
print(f"Pump Accuracy: {pump_accuracy}%")
print(f"Bouncing Up Accuracy: {bouncing_up_accuracy}%")
print(f"Bouncing Down Accuracy: {bouncing_down_accuracy}%")
