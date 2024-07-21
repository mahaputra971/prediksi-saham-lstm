# app/main.py

from pydantic import BaseModel, Field
from typing import List, Any
from datetime import date
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent
from fastapi.staticfiles import StaticFiles
from app.sql import get_table_data
from app.predict import predict_with_loaded_model
from app.exception import exception_handler

from pydantic import BaseModel, Field
from typing import List, Any
from datetime import date
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent, BackEvent, PageEvent
from fastui.forms import fastui_form
from integrations import get_table_data, exception_handler, blob_to_data_url
from fastapi.responses import RedirectResponse
import os
from fastapi.staticfiles import StaticFiles
import json


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class EmitenForm(BaseModel):
    emiten_name: str = Field(title="Emiten Code")

class DateRangeForm(BaseModel):
    start_date: date = Field(title="Start Date")
    end_date: date = Field(title="End Date")

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
                c.Button(text='Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Error Metrics', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Charts', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
    ]

@exception_handler
@app.get("/api/predict_by_date/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
async def predict_by_date(emiten_name: str) -> List[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Heading(text=f'Predict by Date for {emiten_name}', level=2),
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
                    c.Heading(text=f'Prediction Error for {emiten_name}', level=2),
                    c.Paragraph(text='There was an error in processing your prediction. Please check the input data and try again.'),
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

@app.get("/api/navigation/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def navigation(emiten_name: str) -> List[Any]:
    return [
        c.Page(
            components=[
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
                c.Heading(text='Select Action for Emiten', level=2),
                c.Button(text='Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Error Metrics', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Charts', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
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
    image_path_accuracy = f"/static/accuracy/{emiten_name}.png"
    image_path_adj_closing = f"/static/adj_closing_price/{emiten_name}.png"
    image_path_close_price = f"/static/close_price_history/{emiten_name}.png"
    image_path_ichimoku = f"/static/ichimoku/{emiten_name}.png"
    image_path_prediction = f"/static/prediction/{emiten_name}.png"
    image_path_sales_volume = f"/static/sales_volume/{emiten_name}.png"
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

@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))
