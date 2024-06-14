
import pandas as pd
from fastapi import FastAPI
from ..sql import show_specific_tables, get_issuer
from ..ic_project import ichimoku_project
from pydantic import BaseModel


app = FastAPI()
    
class StockValueResponse(BaseModel):
    kode_emiten: str
    close_price: float
    tenkan_sen: dict
    kijun_sen: dict
    senkou_span: dict
    tenkan_sen_status : str
    kijun_sen_status : str
    senkou_span_status : str
    date: str

class StockPriceResponse(BaseModel):
    kode_emiten: str
    open: float
    high: float
    low: float
    close: float
    close_adj: float
    volume: int
    date: str

class ErrorMetricsResponse(BaseModel):
    kode_emiten: str
    RMSE: float
    MAPE: float
    MAE: float
    MSE: float
    date: str

class ChartResponse(BaseModel):
    kode_emiten: str
    pic_closing_price: str
    pic_sales_volume: str
    pic_price_history: str
    pic_comparation: str
    pic_prediction: str
    pic_ichimoku_cloud: str
    render_date: str

@app.get("/stock/value/{issuer_stock_code}", response_model=StockValueResponse)
def get_stock_value(issuer_stock_code: str):
    # issuer = get_issuer()
    # data = ichimoku_project(issuer[0])
    data = ichimoku_project("issuer_stock_code")

    if data is None or data.empty:
        return {"error": "Data not available"}

    latest_row = data.iloc[-1]

    response = {
        "kode_emiten": issuer_stock_code,
        "close_price": latest_row["close"],
        "tenkan_sen": {
            "value": latest_row["tenkan_sen"],
            "status": latest_row["tenkan_sen_status"]
        },
        "kijun_sen": {
            "value": latest_row["kijun_sen"],
            "status": latest_row["kijun_sen_status"]
        },
        "senkou_span": {
            "value_span_a": latest_row["senkou_span_a"],
            "value_span_b": latest_row["senkou_span_b"],
            "status": latest_row["senkou_span_status"]
        },
        "date": latest_row["Date"].strftime('%Y-%m-%d')
    }

    return response

@app.get("/stock/price/{issuer_stock_code}", response_model=StockPriceResponse)
def get_stock_price(issuer_stock_code: str):
    data = show_specific_tables("your_table_name")  # Replace with your table name
    df = pd.DataFrame(data)

    if df.empty:
        return {"error": "Data not available"}

    latest_row = df.iloc[-1]

    response = {
        "kode_emiten": issuer_stock_code,
        "open": latest_row["open"],
        "high": latest_row["high"],
        "low": latest_row["low"],
        "close": latest_row["close"],
        "close_adj": latest_row["close_adj"],
        "volume": latest_row["volume"],
        "date": latest_row["date"].strftime('%Y-%m-%d')
    }

    return response

@app.get("/stock/error-metrics/{issuer_stock_code}", response_model=ErrorMetricsResponse)
def get_error_metrics(issuer_stock_code: str):
    data = show_specific_tables("your_table_name")  # Replace with your table name
    df = pd.DataFrame(data)

    if df.empty:
        return {"error": "Data not available"}

    latest_row = df.iloc[-1]

    response = {
        "kode_emiten": issuer_stock_code,
        "RMSE": latest_row["RMSE"],
        "MAPE": latest_row["MAPE"],
        "MAE": latest_row["MAE"],
        "MSE": latest_row["MSE"],
        "date": latest_row["date"].strftime('%Y-%m-%d')
    }

    return response

@app.get("/stock/chart/{issuer_stock_code}", response_model=ChartResponse)
def get_chart(issuer_stock_code: str):
    data = show_specific_tables("your_table_name")  # Replace with your table name
    df = pd.DataFrame(data)

    if df.empty:
        return {"error": "Data not available"}

    latest_row = df.iloc[-1]

    response = {
        "kode_emiten": issuer_stock_code,
        "pic_closing_price": latest_row["pic_closing_price"],
        "pic_sales_volume": latest_row["pic_sales_volume"],
        "pic_price_history": latest_row["pic_price_history"],
        "pic_comparation": latest_row["pic_comparation"],
        "pic_prediction": latest_row["pic_prediction"],
        "pic_ichimoku_cloud": latest_row["pic_ichimoku_cloud"],
        "render_date": latest_row["render_date"].strftime('%Y-%m-%d')
    }

    return response


# from typing import Union
# from pandas_datareader import data as pdr
# import yfinance as yf
# import datetime

# import matplotlib.pyplot as plt
# import pandas_datareader.data as wb 

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.get("/stock/value/{issuer_stock_codes}") # route {url}/stock/value/BYAN.JK
# def factor_ichimoku_cloud(issuer_stock_codes: str ):
#     # issuer_stock_codes = 'BYAN.JK' 
#     yf.pdr_override()
#     end_date = datetime.datetime.now()
#     start_date = end_date - datetime.timedelta(days=5*365)
#     data = pdr.get_data_yahoo(issuer_stock_codes, start=start_date, end=end_date.strftime("%Y-%m-%d"))
#     high9 = data.High.rolling(9).max()
#     Low9 = data.High.rolling(9).min()   
#     high26 = data.High.rolling(26).max()
#     Low26 = data.High.rolling(26).min()
#     high52 = data.High.rolling(52).max()
#     Low52 = data.High.rolling(52).min()

#     data['tenkan_sen'] = (high9 + Low9) / 2
#     data['kijun_sen'] = (high26 + Low26) / 2
#     data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26) 
#     data['senkou_span_b'] = ((high52 + Low52) / 2).shift(26)
#     data['chikou'] = data.Close.shift(-26)
#     data = data.iloc[26:]

#     print ("\nKode Emiten : ", issuer_stock_codes)
#     print("Close Price:", data['Close'].iloc[-1])
#     print("tenkan_sen:", data['tenkan_sen'].iloc[-1])
#     print("kijun_sen:", data['kijun_sen'].iloc[-1])
#     print("senkou_span_a:", data['senkou_span_a'].iloc[-1])
#     print("senkou_span_b:", data['senkou_span_b'].iloc[-1])
#     tenkan_sen_response = tenkan_sen_factor(data)
#     kijun_sen_response = kijun_sen_factor(data)
#     senkou_span_response = senkou_span_factor(data)
    
#     data = {
#     "kode_emiten": issuer_stock_codes,
#     "close_price": data['Close'].iloc[-1],
#     "tenkan_sen" : {
#         "value": data['tenkan_sen'].iloc[-1],
#         "status": tenkan_sen_response['message_tenkan']
#     },
#     "kijun_sen" : {
#         "value": data['kijun_sen'].iloc[-1],
#         "status": kijun_sen_response["message_kijun"]
#     },
#     "senkou_span" : {
#         "value_span_a": data['senkou_span_a'].iloc[-1],
#         "value_span_b": data['senkou_span_b'].iloc[-1],
#         "status": senkou_span_response["message_senkou_span"]
#     }
# }

#     return data
#     # kijun_sen_factor(data)
#     # senkou_span_factor(data)

# def tenkan_sen_factor(data):
#     tenkan_sen = data['tenkan_sen']
#     # message_tenkan = data['message_tenkan']
#     x = np.array(range(len(tenkan_sen))).reshape(-1, 1)
#     y = tenkan_sen.values.reshape(-1, 1)
#     model = LinearRegression()
#     model.fit(x, y)
#     slope = model.coef_[0]
#     if slope > 0:
#         print("The Tenkan-Sen is in an uptrend.")
#         return {"message_tenkan": "The Tenkan-Sen is in an uptrend."}
#     elif slope < 0:
#         print("The Tenkan-Sen is in an uptrend.")
#         return {"message_tenkan": "The Tenkan-Sen is in an uptrend."}
#     else:
#         print("The Tenkan-Sen is moving sideways.")
#         return {"message_tenkan": "The Tenkan-Sen is moving sideways."}
        
# def kijun_sen_factor(data):
#     last_close = data['Close'].iloc[-1]
#     last_kijun_sen = data['kijun_sen'].iloc[-1]

#     if last_close > last_kijun_sen:
#         print("The kijun sen is in an upward trend.")
#         return {"message_kijun": "The kijun sen is in an upward trend."}
#     elif last_close < last_kijun_sen:
#         print("The kijun sen is in a downward trend.")
#         return {"message_kijun": "The kijun sen is in a downward trend."}
#     else:
#         print("The kijun sen is moving sideways.")
#         return {"message_kijun": "The kijun sen is moving sideways."}
        
# def senkou_span_factor(data):
#     last_close = data['Close'].iloc[-1]
#     last_senkou_span_a = data['senkou_span_a'].iloc[-1]
#     last_senkou_span_b = data['senkou_span_b'].iloc[-1]

#     if last_close > last_senkou_span_a and last_senkou_span_a > last_senkou_span_b:
#         print("Status senkou span: Uptrend")
#         return {"message_senkou_span": "Status senkou span: Uptrend"}
#     elif last_close < last_senkou_span_a and last_senkou_span_a < last_senkou_span_b:
#         print("Status senkou span: Downtrend")
#         return {"message_senkou_span": "Status senkou span: Downtrend"}
#     elif last_close < last_senkou_span_b and last_senkou_span_a > last_senkou_span_b:
#         print("Status senkou span: Will Dump")
#         return {"message_senkou_span": "Status senkou span: Will Dump"}
#     elif last_close > last_senkou_span_b and last_senkou_span_a < last_senkou_span_b:
#         print("Status senkou span: Will Pump")
#         return {"message_senkou_span": "Status senkou span: Will Pump"}
#     elif last_senkou_span_b < last_close < last_senkou_span_a and last_senkou_span_a > last_senkou_span_b:
#         print("Status senkou span: Uptrend and Will Bounce Up")
#         return {"message_senkou_span": "Status senkou span: Uptrend and Will Bounce Up"}
#     elif last_senkou_span_b < last_close < last_senkou_span_a and last_senkou_span_a < last_senkou_span_b:
#         print("Status senkou span): Downtrend and Will Bounce Down")
#         return {"message_senkou_span": "Status senkou span): Downtrend and Will Bounce Down"}

# # @app.get("/stock/prediction/{item_id}")