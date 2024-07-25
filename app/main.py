# app/main.py

from pydantic import BaseModel, Field, field_validator, ValidationError, ValidationInfo
from typing import List, Any
from datetime import date
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent
from fastapi.staticfiles import StaticFiles
from app.sql import get_table_data
from app.predict import predict_with_loaded_model, predict_future, ichimoku_predict, train_and_evaluate_model
from app.exception import exception_handler


from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import FileResponse
from fastui.events import BackEvent, PageEvent
from fastui.forms import fastui_form
from integrations import exception_handler, blob_to_data_url
import os
from fastapi.staticfiles import StaticFiles
import json
from datetime import datetime, timedelta
import yfinance as yf


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/static2", StaticFiles(directory="picture"), name="static")

class EmitenForm(BaseModel):
    emiten_name: str = Field(title="Emiten Code")

class DateRangeForm(BaseModel):
    start_date: date = Field(title="Start Date")
    end_date: date = Field(title="End Date")

class DateEndRangeForm(BaseModel):
    date: int = Field(title="Future Days")

    @field_validator('date')
    def check_date(cls, v, info: ValidationInfo):
        emiten_name = info.context.get('emiten_name')
        if not emiten_name:
            raise ValueError("Emiten name is required for validation")
        
        # Download stock data to determine the available historical data length
        data = yf.download(emiten_name)
        if data.empty:
            raise ValueError(f"No historical data found for {emiten_name}")
        
        available_days = len(data)
        if v > available_days - 60:
            raise ValueError(f"The value must be less than or equal to {available_days - 60}")
        return v

class IchimokuForm(BaseModel):
    specific_date: date = Field(title="Specific Date")

    @field_validator('specific_date')
    def check_specific_date(cls, v, info: ValidationInfo):
        emiten_name = info.context.get('emiten_name')
        if not emiten_name:
            raise ValueError("Emiten name is required for validation")
        
        # Download stock data to determine the latest available date
        data = yf.download(emiten_name)
        if data.empty:
            raise ValueError(f"No historical data found for {emiten_name}")
        
        latest_date = data.index.max().date()
        earliest_date = data.index.min().date()
        if v > latest_date:
            raise ValueError(f"The date {v} is beyond the latest available date {latest_date} in the historical data for {emiten_name}")
        if v < earliest_date:
            raise ValueError(f"The date {v} is less than the earliest available date {earliest_date} in the historical data for {emiten_name}")
        return v

class LSTMForm(BaseModel):
    start_date: date = Field(title="Start Date")
    end_date: date = Field(title="End Date")
    future_date: date = Field(title="Future Date")

    @field_validator('start_date')
    def check_start_date(cls, v, info: ValidationInfo):
        emiten_name = info.context.get('emiten_name')
        if not emiten_name:
            raise ValueError("Emiten name is required for validation")
        
        data = yf.download(emiten_name)
        if data.empty:
            raise ValueError(f"No historical data found for {emiten_name}")
        
        earliest_date = data.index.min().date()
        if v < earliest_date:
            raise ValueError(f"The start date {v} is earlier than the earliest available data date {earliest_date}")
        return v

    @field_validator('end_date')
    def check_end_date(cls, v, info: ValidationInfo):
        if 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError("End date must be after the start date")
        return v

    @field_validator('future_date')
    def check_future_date(cls, v, info: ValidationInfo):
        emiten_name = info.context.get('emiten_name')
        if not emiten_name:
            raise ValueError("Emiten name is required for validation")

        data = yf.download(emiten_name)
        if data.empty:
            raise ValueError(f"No historical data found for {emiten_name}")

        latest_date = data.index.max().date()
        if v > latest_date:
            raise ValueError(f"The future date {v} is beyond the latest available data date {latest_date}")
        if (latest_date - timedelta(days=60)) < info.data['end_date']:
            raise ValueError(f"End date must be at least 60 days before the latest data date {latest_date}")
        return v
    
class IchimokuData(BaseModel):
    kode_emiten: str
    tenkan_sen: int
    kijun_sen: int
    senkou_span_a: int
    senkou_span_b: int
    date: str

class StockValueResponse(BaseModel):
    kode_emiten: str
    close_price: float
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    tenkan_sen_status: str
    kijun_sen_status: str
    senkou_span_status: str
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

class IchimokuStatus(BaseModel):
    kode_emiten: str
    sen_status: str
    span_status: str
    date: date

class PredictionLSTM(BaseModel):
    kode_emiten: str
    max_price: float
    min_price: float
    max_price_date: date
    min_price_date: date
    date: date

class IchimokuAccuracy(BaseModel):
    kode_emiten: str
    percent_1_hari_sen: float
    percent_1_minggu_sen: float
    percent_1_bulan_sen: float
    percent_1_hari_span: float
    percent_1_minggu_span: float
    percent_1_bulan_span: float
    date: date


@exception_handler
@app.get("/")
async def root():
    return RedirectResponse(url="/home")

@exception_handler
@app.get("/api/home", response_model=FastUI, response_model_exclude_none=True)
async def home() -> List[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Heading(text='Analisa Technical LSTM dan Ichimoku Stock', level=2),
                c.ModelForm(model=EmitenForm, display_mode='page', submit_url='/api/submit_emiten_form'),
            ]
        ),
    ]

@exception_handler
@app.post("/api/submit_emiten_form", response_model=FastUI, response_model_exclude_none=True)
async def submit_emiten_form(emiten_name: str = Form(...)):
    return [
        c.Page(
            components=[
                c.Heading(text='Select Action for Emiten', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Result Analyst Data', level=4),
                c.Button(text='Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Error Metrics', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Charts', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Calculate By Yourself', level=4),
                c.Button(text='LSTM by date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='Ichimoku by Date', on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='LSTM by date 2', on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            ]
        )
    ]

@exception_handler
@app.get("/api/predict_by_date/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def predict_by_date(emiten_name: str) -> List[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'Accuracy Predict by Date for {emiten_name}', level=2),
                c.ModelForm(model=DateRangeForm, display_mode='page', submit_url=f'/api/predict_result/{emiten_name}'),
            ]
        ),
    ]

@exception_handler
@app.post("/api/predict_result/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def predict_result(emiten_name: str, start_date: date = Form(...), end_date: date = Form(...)) -> List[AnyComponent]:
    accuracy, plot_url = predict_with_loaded_model(emiten_name, start_date, end_date)
    if accuracy is None:
        return [
            c.Page(
                components=[
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                    c.Heading(text=f'Prediction Error for {emiten_name}', level=2),
                    c.Paragraph(text='There was an error in processing your prediction, make sure not less than 150 days and not fill end date with future date . Please check the input data and try again.'),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}')),
                ]
            ),
        ]

    return [
        c.Page(
            components=[
                c.Heading(text=f'Prediction Accuracy for {emiten_name}', level=2),
                c.Paragraph(text=f'Accuracy: {accuracy}'),
                c.Image(src=plot_url, alt='Prediction Plot', width=1000, height=500, loading='lazy', referrer_policy='no-referrer', class_name='border rounded'),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}')),
            ]
        ),
    ]
    
@exception_handler
@app.get("/api/predict_price_by_date/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def predict_price_by_date(emiten_name: str) -> List[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'Predict Price by Date for {emiten_name}', level=2),
                c.ModelForm(model=DateEndRangeForm, display_mode='page', submit_url=f'/api/predict_price/{emiten_name}'),
            ]
        ),
    ]

@app.post("/api/predict_price/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def predict_price(emiten_name: str, date: int = Form(...)) -> List[AnyComponent]:
    try:
        # Validasi input dengan menyuntikkan emiten_name ke dalam context
        form_data = {"date": date}
        validator_context = {"emiten_name": emiten_name}
        DateEndRangeForm.model_validate(form_data, context=validator_context)
    except ValidationError as e:
        return [
            c.Page(
                components=[
                    c.Heading(text='Input Error', level=2),
                    c.Paragraph(text=str(e)),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}')),
                ]
            )
        ]
    
    # Lanjutkan dengan prediksi jika input valid
    predictions, plot_url = predict_future(emiten_name, date)
    if predictions is None:
        return [
            c.Page(
                components=[
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                    c.Heading(text=f'Prediction Error for {emiten_name}', level=2),
                    c.Paragraph(text='There was an error in processing your prediction. Please check the input data and try again.'),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}')),
                ]
            ),
        ]

    highest_price, lowest_price, max_price_date, min_price_date = predictions

    return [
        c.Page(
            components=[
                c.Heading(text=f'Prediction for {emiten_name}', level=2),
                c.Paragraph(text=f'Highest Price: {highest_price} on {max_price_date}'),
                c.Paragraph(text=f'Lowest Price: {lowest_price} on {min_price_date}'),
                c.Image(src=plot_url, alt='Prediction Plot', width=1000, height=500, loading='lazy', referrer_policy='no-referrer', class_name='border rounded'),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}')),
            ]
        ),
    ]

@app.get("/api/navigation/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def navigation(emiten_name: str) -> List[Any]:
    return [
        c.Page(
            components=[
                c.Heading(text='Select Action for Emiten', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Result Analyst Data', level=4),
                c.Button(text='Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Error Metrics', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Charts', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Calculate By Yourself', level=4),
                c.Button(text='LSTM by date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='Ichimoku by Date', on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='LSTM by date 2', on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            ]
        )
    ]

@exception_handler
@app.get("/api/detail_emiten/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def detail_emiten_table(emiten_name: str) -> List[Any]:
    detail_emiten = get_table_data(emiten_name, 'tb_detail_emiten')
    detail_emiten = [StockPriceResponse(**{**item, 'kode_emiten': emiten_name}) for item in detail_emiten]
    return [
        c.Page(
            components=[
                c.Heading(text='Detail Emiten', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=detail_emiten,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='open'),
                        DisplayLookup(field='high'),
                        DisplayLookup(field='low'),
                        DisplayLookup(field='close'),
                        DisplayLookup(field='close_adj'),
                        DisplayLookup(field='volume'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@exception_handler
@app.get("/api/ichimoku_data/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def ichimoku_data_table(emiten_name: str) -> List[Any]:
    ichimoku_data = get_table_data(emiten_name, 'tb_data_ichimoku_cloud')
    ichimoku_data = [IchimokuData(**{**item, 'kode_emiten': emiten_name}) for item in ichimoku_data]
    return [
        c.Page(
            components=[
                c.Heading(text='Ichimoku Data', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=ichimoku_data,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='tenkan_sen'),
                        DisplayLookup(field='kijun_sen'),
                        DisplayLookup(field='senkou_span_a'),
                        DisplayLookup(field='senkou_span_b'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@exception_handler
@app.get("/api/error_metrics/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def error_metrics_table(emiten_name: str) -> List[Any]:
    lstm_data = get_table_data(emiten_name, 'tb_lstm')
    lstm_data = [ErrorMetricsResponse(**{**item, 'kode_emiten': emiten_name}) for item in lstm_data]
    return [
        c.Page(
            components=[
                c.Heading(text='Error Metrics', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=lstm_data,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='RMSE'),
                        DisplayLookup(field='MAPE'),
                        DisplayLookup(field='MAE'),
                        DisplayLookup(field='MSE'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@exception_handler
@app.get("/api/prediction/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def error_metrics_table(emiten_name: str) -> List[Any]:
    tb_prediction_lstm = get_table_data(emiten_name, 'tb_prediction_lstm')
    tb_prediction_lstm = [PredictionLSTM(**{**item, 'kode_emiten': emiten_name}) for item in tb_prediction_lstm]
    return [
        c.Page(
            components=[
                c.Heading(text='Error Metrics', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=tb_prediction_lstm,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='max_price'),
                        DisplayLookup(field='min_price'),
                        DisplayLookup(field='max_price_date'),
                        DisplayLookup(field='min_price_date'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@exception_handler
@app.get("/api/ichimoku_status/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def ichimoku_status_table(emiten_name: str) -> List[Any]:
    ichimoku_status_data = get_table_data(emiten_name, 'tb_ichimoku_status')
    ichimoku_status_data = [IchimokuStatus(**{**item, 'kode_emiten': emiten_name}) for item in ichimoku_status_data]
    return [
        c.Page(
            components=[
                c.Heading(text='Ichimoku Status', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=ichimoku_status_data,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='sen_status'),
                        DisplayLookup(field='span_status'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@app.get("/api/charts/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def charts_table(emiten_name: str) -> List[Any]:
    emiten_chart = get_table_data(emiten_name, 'tb_summary')
    image_path_accuracy = f"/static2/accuracy/{emiten_name}.png"
    image_path_adj_closing = f"/static2/adj_closing_price/{emiten_name}.png"
    image_path_close_price = f"/static2/close_price_history/{emiten_name}.png"
    image_path_ichimoku = f"/static2/ichimoku/{emiten_name}.png"
    image_path_prediction = f"/static2/prediction/{emiten_name}.png"
    image_path_sales_volume = f"/static2/sales_volume/{emiten_name}.png"
    emiten_chart = [ChartResponse(**{**item, 'kode_emiten': emiten_name}) for item in emiten_chart]
    return [
        c.Page(
            components=[
                c.Heading(text='Charts', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=emiten_chart,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='pic_closing_price'),
                        DisplayLookup(field='pic_sales_volume'),
                        DisplayLookup(field='pic_price_history'),
                        DisplayLookup(field='pic_comparation'),
                        DisplayLookup(field='pic_prediction'),
                        DisplayLookup(field='pic_ichimoku_cloud'),
                        DisplayLookup(field='render_date'),
                    ],
                ),
            ]
        ),
        c.Div(
            components=[
                c.Heading(text='Accuracy LSTM', level=2),
                c.Paragraph(text='This shows how accurate the LSTM model is.'),
                c.Image(
                    src=image_path_accuracy,
                    alt='Accuracy LSTM',
                    width=1000,
                    height=500,
                    loading='lazy',
                    referrer_policy='no-referrer',
                    class_name='border rounded',
                ),
            ],
            class_name='border-top mt-3 pt-1 center-content',
        ),
        c.Div(
            components=[
                c.Heading(text='Adjusted Close Price', level=2),
                c.Paragraph(text='This shows the adjusted close price of the stock.'),
                c.Image(
                    src=image_path_adj_closing,
                    alt='Adjusted Close Price',
                    width=1000,
                    height=500,
                    loading='lazy',
                    referrer_policy='no-referrer',
                    class_name='border rounded',
                ),
            ],
            class_name='border-top mt-3 pt-1 center-content',
        ),
        c.Div(
            components=[
                c.Heading(text='Close Price History', level=2),
                c.Paragraph(text='This shows the close price history of the stock.'),
                c.Image(
                    src=image_path_close_price,
                    alt='Close Price History',
                    width=1000,
                    height=500,
                    loading='lazy',
                    referrer_policy='no-referrer',
                    class_name='border rounded',
                ),
            ],
            class_name='border-top mt-3 pt-1 center-content',
        ),
        c.Div(
            components=[
                c.Heading(text='Ichimoku Cloud', level=2),
                c.Paragraph(text='This shows the ichimoku cloud of the stock.'),
                c.Image(
                    src=image_path_ichimoku,
                    alt='Ichimoku Cloud',
                    width=1000,
                    height=500,
                    loading='lazy',
                    referrer_policy='no-referrer',
                    class_name='border rounded',
                ),
            ],
            class_name='border-top mt-3 pt-1 center-content',
        ),
        c.Div(
            components=[
                c.Heading(text='Prediction', level=2),
                c.Paragraph(text='This shows the predicted close price of the stock.'),
                c.Image(
                    src=image_path_prediction,
                    alt='Prediction',
                    width=1000,
                    height=500,
                    loading='lazy',
                    referrer_policy='no-referrer',
                    class_name='border rounded',
                ),
            ],
            class_name='border-top mt-3 pt-1 center-content',
        ), 
        c.Div(
            components=[
                c.Heading(text='Sales Volume', level=2),
                c.Paragraph(text='This shows the sales volume of the stock.'),
                c.Image(
                    src=image_path_sales_volume,
                    alt='Sales Volume',
                    width=1000,
                    height=500,
                    loading='lazy',
                    referrer_policy='no-referrer',
                    class_name='border rounded',
                ),
            ],
            class_name='border-top mt-3 pt-1 center-content',
        ),     
    ]

@exception_handler
@app.get("/api/ichimoku_accuracy/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def ichimoku_status_table(emiten_name: str) -> List[Any]:
    accuracy_ichimoku_cloud = get_table_data(emiten_name, 'tb_accuracy_ichimoku_cloud')
    accuracy_ichimoku_cloud = [IchimokuAccuracy(**{**item, 'kode_emiten': emiten_name}) for item in accuracy_ichimoku_cloud]
    return [
        c.Page(
            components=[
                c.Heading(text='Ichimoku Accuracy', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=accuracy_ichimoku_cloud,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='percent_1_hari_sen'),
                        DisplayLookup(field='percent_1_minggu_sen'),
                        DisplayLookup(field='percent_1_bulan_sen'),
                        DisplayLookup(field='percent_1_hari_span'),
                        DisplayLookup(field='percent_1_minggu_span'),
                        DisplayLookup(field='percent_1_bulan_span'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@exception_handler
@app.get("/api/ichimoku_by_date/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def ichimoku_by_date(emiten_name: str) -> List[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'Ichimoku Cloud Status by Date for {emiten_name}', level=2),
                c.ModelForm(model=IchimokuForm, display_mode='page', submit_url=f'/api/ichimoku_by_date_result/{emiten_name}'),
            ]
        ),
    ]
    
@exception_handler
@app.post("/api/ichimoku_by_date_result/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def ichimoku_by_date_result(emiten_name: str, specific_date: date = Form(...)) -> List[AnyComponent]:
    try:
        # Validasi input dengan menyuntikkan emiten_name ke dalam context
        form_data = {"specific_date": specific_date}
        validator_context = {"emiten_name": emiten_name}
        IchimokuForm.model_validate(form_data, context=validator_context)
    except ValidationError as e:
        return [
            c.Page(
                components=[
                    c.Heading(text='Input Error', level=2),
                    c.Paragraph(text=str(e)),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}')),
                ]
            )
        ]
    
    # Lanjutkan dengan prediksi jika input valid
    span_status, sen_status = ichimoku_predict(emiten_name, specific_date)
    if span_status is None or sen_status is None:
        return [
            c.Page(
                components=[
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                    c.Heading(text=f'Prediction Error for {emiten_name}', level=2),
                    c.Paragraph(text='There was an error in processing your prediction. Please check the input data and try again.'),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}')),
                ]
            ),
        ]

    return [
        c.Page(
            components=[
                c.Heading(text=f'Span Status: {span_status}', level=3),
                c.Heading(text=f'Sen Status: {sen_status}', level=3),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}')),
            ]
        ),
    ]

#LSTM    
@exception_handler
@app.get("/api/lstm_accuracy/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def ichimoku_by_date(emiten_name: str) -> List[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'LSTM by Date for {emiten_name}', level=2),
                c.ModelForm(model=LSTMForm, display_mode='page', submit_url=f'/api/lstm_accuracy_result/{emiten_name}'),
            ]
        ),
    ]

@exception_handler
@app.post("/api/lstm_accuracy_result/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def lstm_accuracy_result(emiten_name: str, start_date: date = Form(...), end_date: date = Form(...), future_date: date = Form(...)) -> List[AnyComponent]:
    try:
        # Validasi input dengan konteks emiten_name
        form_data = {"start_date": start_date, "end_date": end_date, "future_date": future_date}
        LSTMForm.model_validate(form_data, context={"emiten_name": emiten_name})
    except ValidationError as e:
        return [
            c.Page(
                components=[
                    c.Heading(text='Input Error', level=2),
                    c.Paragraph(text=str(e)),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}')),
                ]
            )
        ]
    
    try:
        # Prediksi menggunakan model
        result, plot_path = train_and_evaluate_model(emiten_name, start_date, end_date, future_date)
    except ValueError as e:
        return [
            c.Page(
                components=[
                    c.Heading(text='Prediction Error', level=2),
                    c.Paragraph(text=str(e)),
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}')),
                ]
            )
        ]
        
    mae, mse, rmse, mape = result

    return [
        c.Page(
            components=[
                c.Heading(text=f'Prediction Results for {emiten_name}', level=2),
                c.Paragraph(text=f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}%'),
                c.Image(src=plot_path, alt='LSTM Plot', width=1000, height=500, loading='lazy', referrer_policy='no-referrer', class_name='border rounded'),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}')),
            ]
        )
    ]
    
@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))
