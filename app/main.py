# app/main.py

from pydantic import BaseModel, Field, field_validator, ValidationError, ValidationInfo
import pandas as pd
from typing import List, Any
from datetime import date
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent
from fastapi.staticfiles import StaticFiles
from app.sql import get_table_data, get_emiten_status, get_emiten_id, insert_data_analyst, fetch_emiten_recommendation
from app.predict import predict_with_loaded_model, forcasting_stock, forcasting_stock2, ichimoku_predict, train_and_evaluate_model2, train_and_evaluate_model3
from app.exception import exception_handler
from app.engine import engine_main, train_and_evaluate_model, predict_future

from fastapi import FastAPI, HTTPException, Request, Form   
from fastapi.responses import FileResponse
from fastui.events import BackEvent, PageEvent
from fastui.forms import fastui_form
from fastapi import Query
from integrations import exception_handler, blob_to_data_url, ichimoku_sql, pembuktian_ichimoku
import os
from fastapi.staticfiles import StaticFiles
import json
from datetime import datetime, timedelta
import yfinance as yf

app = FastAPI()
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
# app.mount("/static2", StaticFiles(directory="picture"), name="static")

# Menggunakan path absolut
app.mount("/static", StaticFiles(directory=os.path.abspath("app/static")), name="static")
app.mount("/static2", StaticFiles(directory=os.path.abspath("picture")), name="static")

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
    
    @staticmethod
    def check_date_differences(start_date, end_date, future_date):
        start_end_diff = (end_date - start_date).days
        if start_end_diff <= 60:
            raise ValueError("The difference between start date and end date must be more than 60 days")
        end_future_diff = (future_date - end_date).days
        if start_end_diff <= end_future_diff:
            raise ValueError("The difference between start date and end date must be greater than the difference between end date and future date")

    @field_validator('start_date')
    def check_start_date(cls, v, info: ValidationInfo):
        if 'end_date' in info.data and 'future_date' in info.data:
            cls.check_date_differences(v, info.data['end_date'], info.data['future_date'])
        return v

    @field_validator('end_date')
    def check_end_date(cls, v, info: ValidationInfo):
        if 'start_date' in info.data and 'future_date' in info.data:
            cls.check_date_differences(info.data['start_date'], v, info.data['future_date'])
        return v

    @field_validator('future_date')
    def check_future_date(cls, v, info: ValidationInfo):
        if 'start_date' in info.data and 'end_date' in info.data:
            cls.check_date_differences(info.data['start_date'], info.data['end_date'], v)
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
    accuracy: float
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
    max_price_date: date
    min_price: float
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
    
class PredicValid(BaseModel):
    kode_emiten: str 
    Predictions: float
    Close: float
    Date: date
    
class LSTMAccuracy(BaseModel): 
    kode_emiten: str 
    mean_gap: float 
    highest_gap: float 
    lowest_gap: float
    date: date
    
class PredictionPriceResponse(BaseModel):
    price: float
    date: date
    kode_emiten: str
    
class Recommendation(BaseModel): 
    kode_emiten: str
    date: date

@exception_handler
@app.get("/")
async def root():
    return RedirectResponse(url="/home")

@exception_handler
@app.get("/return")
async def returnx():
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
        c.Page(
            components=[
                c.Button(text='Recommendation Stock', on_click=GoToEvent(url=f'/recommendation'), named_style='secondary', class_name='ms-2'),
            ]
        ),
    ]

@exception_handler
@app.post("/api/submit_emiten_form", response_model=FastUI, response_model_exclude_none=True)
async def submit_emiten_form(emiten_name: str = Form(...)):
    # Convert emiten_name to uppercase
    emiten_name = emiten_name.upper()
    emiten_name = emiten_name.strip()

    # If emiten_name does not end with ".JK", append ".JK" to it
    if not emiten_name.endswith(".JK"):
        emiten_name += ".JK"
        
    status = get_emiten_status(emiten_name)
    print(status)
    if status == 0:
        try: 
            engine_main(emiten_name)
            return [
                c.Page(
                    components=[    
                        c.Heading(text=f'Select Action for Emiten {emiten_name}', level=2),
                        c.Link(components=[c.Text(text='Back to Home')], on_click=BackEvent()),
                    ]
                ),
                c.Page(
                    components=[
                        c.Heading(text='Result Analyst Data', level=4),
                        c.Heading(text='LSTM', level=6),
                        c.Button(text='Data Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='LSTM Detail', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='Summary', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='Prediction LSTM', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='Accuracy LSTM', on_click=GoToEvent(url=f'/lstm_accuracy_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='Prediction Dump Data', on_click=GoToEvent(url=f'/prediction_price_dump_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                    ]
                ),
                c.Div(
                    components=[
                        c.Heading(text='Ichimoku Cloud', level=6),
                        c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                        c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                    ]
                ),
                c.Page(
                    components=[
                        c.Link(components=[c.Text(text='Moderator')], on_click=GoToEvent(url=f'/navigation_moderator/{emiten_name}')),
                    ]
                ),
                # c.Page(
                #     components=[
                #         c.Heading(text='Calculate By Yourself', level=4),
                #         # c.Button(text='LSTM by date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                #         c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                #         c.Button(text='Ichimoku by Date', on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                #         c.Button(text='LSTM by date 2', on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                #     ]
                # )
            ]
                        
        except ValueError as e:
            return [
                c.Page(
                    components=[
                        c.Heading(text='Prediction Error', level=2),
                        c.Paragraph(text=str(e)),
                        c.Link(components=[c.Text(text='Back')], on_click=BackEvent()),
                    ]
                )
            ]
    elif status == None:
        return [
            c.Page(
                components=[
                    c.Heading(text='Error', level=2),
                    c.Paragraph(text=f'The status for {emiten_name} is wrong and not available on IHSG.'),
                    c.Link(components=[c.Text(text='Back to Home')], on_click=BackEvent()),
                    
                ]
            ),
        ]
    else:
        return [
            c.Page(
                components=[    
                    c.Heading(text=f'Select Action for Emiten {emiten_name}', level=2),
                    c.Link(components=[c.Text(text='Back to Home')], on_click=BackEvent()),
                ]
            ),
            c.Page(
                components=[
                    c.Heading(text='Result Analyst Data', level=4),
                    c.Heading(text='LSTM', level=6),
                    c.Button(text='Data Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='LSTM Detail', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='Summary', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='Prediction LSTM', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='Accuracy LSTM', on_click=GoToEvent(url=f'/lstm_accuracy_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='Prediction Dump Data', on_click=GoToEvent(url=f'/prediction_price_dump_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                ]
            ),
            c.Div(
                components=[
                    c.Heading(text='Ichimoku Cloud', level=6),
                    c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                    c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                ]
            ),
            c.Page(
                    components=[
                        c.Link(components=[c.Text(text='Moderator')], on_click=GoToEvent(url=f'/navigation_moderator/{emiten_name}')),
                    ]
            ),
            # c.Page(
            #     components=[
            #         c.Heading(text='Calculate By Yourself', level=4),
            #         # c.Button(text='LSTM by date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            #         c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            #         c.Button(text='Ichimoku by Date', on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            #         c.Button(text='LSTM by date 2', on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            #     ]
            # )
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
    predictions, plot_url = forcasting_stock(emiten_name, date)
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
                c.Heading(text=f'Select Action for Emiten {emiten_name}', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Result Analyst Data', level=4),
                c.Heading(text='LSTM', level=6),
                c.Button(text='Data Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                c.Button(text='LSTM Detail', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Summary', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction LSTM', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Accuracy LSTM', on_click=GoToEvent(url=f'/lstm_accuracy_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction Dump Data', on_click=GoToEvent(url=f'/prediction_price_dump_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
            ]
        ),  
        c.Page(
            components=[
                c.Heading(text='Ichimoku Cloud', level=6),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Rekomendasi', level=6),
                c.Button(text='Recommendation Stock', on_click=GoToEvent(url=f'/recommendation'), named_style='secondary', class_name='ms-2'),
            ]
        ),
        c.Page(
            components=[
                c.Link(components=[c.Text(text='Moderator')], on_click=GoToEvent(url=f'/navigation_moderator/{emiten_name}')),
            ]
        ),
        # c.Page(
        #     components=[
        #         c.Heading(text='Calculate By Yourself', level=4),
        #         # c.Button(text='LSTM by date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
        #         c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
        #         c.Button(text='Ichimoku by Date', on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
        #         c.Button(text='LSTM by date 2', on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
        #     ]
        # )
    ]
    
@app.get("/api/navigation_moderator/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def navigation(emiten_name: str) -> List[Any]:
    return [
        c.Page(
            components=[
                c.Heading(text=f'Select Action for Emiten {emiten_name}', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Result Analyst Data', level=4),
                c.Heading(text='LSTM', level=6),
                c.Button(text='Data Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                c.Button(text='LSTM Detail', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Summary', on_click=GoToEvent(url=f'/charts/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction LSTM', on_click=GoToEvent(url=f'/prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Accuracy LSTM', on_click=GoToEvent(url=f'/lstm_accuracy_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Prediction Dump Data', on_click=GoToEvent(url=f'/prediction_price_dump_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
            ]
        ),  
        c.Page(
            components=[
                c.Heading(text='Ichimoku Cloud', level=6),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}/1'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Status', on_click=GoToEvent(url=f'/ichimoku_status/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Button(text='Ichimoku Accuracy', on_click=GoToEvent(url=f'/ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Calculate By Yourself', level=4),
                # c.Button(text='LSTM by date', on_click=GoToEvent(url=f'/predict_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='Predict by Date', on_click=GoToEvent(url=f'/predict_price_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='Ichimoku by Date', on_click=GoToEvent(url=f'/ichimoku_by_date/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
                c.Button(text='LSTM by date 2', on_click=GoToEvent(url=f'/lstm_accuracy/{emiten_name}'), named_style='warning', class_name='+ ms-2'),
            ]
        ),
        c.Page(
            components=[
                c.Heading(text='Rekomendasi', level=6),
                c.Button(text='Recommendation Stock', on_click=GoToEvent(url=f'/recommendation'), named_style='secondary', class_name='ms-2'),
            ]
        ),
    ]

@exception_handler
@app.get("/api/detail_emiten_next/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def detail_emiten_next_page(emiten_name: str, current_page: int):
    current_page = current_page + 1
    return RedirectResponse(url=f"/api/detail_emiten/{emiten_name}/{current_page}")

@exception_handler
@app.get("/api/detail_emiten_prev/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def detail_emiten_prev_page(emiten_name: str, current_page: int) -> RedirectResponse:
    return RedirectResponse(url=f"/api/detail_emiten/{emiten_name}/{max(1, current_page - 1)}")

@exception_handler
@app.get("/api/detail_emiten/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def detail_emiten_table(emiten_name: str, current_page: int = 1, page_size: int = 10) -> List[Any]:
    detail_emiten = get_table_data(emiten_name, 'tb_detail_emiten')
    detail_emiten = [StockPriceResponse(**{**item, 'kode_emiten': emiten_name}) for item in detail_emiten]

    # Convert the 'date' field to datetime objects if they are strings
    for item in detail_emiten:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in detail_emiten)
    latest_date = max(item.date for item in detail_emiten)

    # Run This if i want to click update button
    # update_detail_emiten(latest_date, emiten_name, detail_emiten)

    # Convert 'date' fields back to strings if needed before returning
    for item in detail_emiten:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')
            
    total_pages = len(detail_emiten) // page_size
    if len(detail_emiten) % page_size > 0:
        total_pages += 1

    # Pagination
    start = (current_page - 1) * page_size
    end = start + page_size
    detail_emiten = detail_emiten[start:end]

    components_with_next = [
        c.Heading(text=f'Detail Emiten {emiten_name}', level=2),
        c.Heading(text=f'From {earliest_date.strftime('%Y-%m-%d')} To {latest_date.strftime('%Y-%m-%d')}', level=6),
        c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
        c.Button(text='Previous', on_click=GoToEvent(url=f'/detail_emiten_prev/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Button(text='Next', on_click=GoToEvent(url=f'/detail_emiten_next/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Heading(text=f'Deskripsi : Pada page ini berisi data lengkap data yang digunakan untuk melakukan pengolahan data pada prediksi menggunakan LSTM dan teknikal Ichimoku Cloud', level=6),
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
        c.Heading(text=f'Page {current_page}', level=6),
        c.Button(text='Full Data (No Pagination)', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}'), named_style='secondary', class_name='ms-2'),
    ]

    components_without_next = [
        c.Heading(text=f'Detail Emiten {emiten_name}', level=2),
        c.Heading(text=f'From {earliest_date.strftime('%Y-%m-%d')} To {latest_date.strftime('%Y-%m-%d')}', level=6),
        c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
        c.Button(text='Previous', on_click=GoToEvent(url=f'/detail_emiten_prev/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
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
        c.Heading(text=f'Page {current_page}', level=6),
    ]

    return [
        c.Page(
            components=components_with_next if current_page < total_pages else components_without_next
        ),
    ]

@exception_handler
@app.get("/api/detail_emiten/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def detail_emiten_table(emiten_name: str) -> List[Any]:
    detail_emiten = get_table_data(emiten_name, 'tb_detail_emiten')
    detail_emiten = [StockPriceResponse(**{**item, 'kode_emiten': emiten_name}) for item in detail_emiten]

    # Convert the 'date' field to datetime objects if they are strings
    for item in detail_emiten:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in detail_emiten)
    latest_date = max(item.date for item in detail_emiten)

    # Run This if i want to click update button
    # update_detail_emiten(latest_date, emiten_name, detail_emiten)

    # Convert 'date' fields back to strings if needed before returning
    for item in detail_emiten:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')

    return [
        c.Page(
            components=[
                c.Heading(text=f'Detail Emiten {emiten_name}', level=2),
                c.Heading(text=f'From {earliest_date.strftime('%Y-%m-%d')} To {latest_date.strftime('%Y-%m-%d')}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Button(text='Update Data', on_click=GoToEvent(url=f'/update_detail_emiten/{emiten_name}'), named_style='secondary', class_name='ms-2'),
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
    for item in ichimoku_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
    
    for item in ichimoku_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in ichimoku_data)
    latest_date = max(item.date for item in ichimoku_data)
    return [
        c.Page(
            components=[
                c.Heading(text=f'Ichimoku Data {emiten_name}', level=2),
                c.Heading(text=f'From {earliest_date} To {latest_date}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Button(text='Update Data', on_click=GoToEvent(url=f'/update_ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data perhitungan lengkap untuk prediksi teknikal Ichimoku Cloud', level=6),
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
@app.get("/api/update_ichimoku_data/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def update_ichimoku_data(emiten_name: str) -> List[Any]:
    ichimoku_data = get_table_data(emiten_name, 'tb_data_ichimoku_cloud')
    ichimoku_data = [IchimokuData(**{**item, 'kode_emiten': emiten_name}) for item in ichimoku_data]
    for item in ichimoku_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
    
    for item in ichimoku_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')
            
    # Fetch the latest date
    earliest_date = min(item.date for item in ichimoku_data)
    latest_date = max(item.date for item in ichimoku_data)
    if latest_date.date() < datetime.now().date():
        # Download the data from Yahoo Finance for the period between the latest date and the current date
        try:
            data_ic, sen_status, span_status = ichimoku_sql()
            if data_ic.empty:
                print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
            else:
                data_ic = pd.DataFrame(data_ic)
                stock_id = get_emiten_id(emiten_name)
                data_ic['id_emiten'] = stock_id
                insert_data_analyst('tb_data_ichimoku_cloud', data_ic)

        except Exception as e:
            print(f"Failed to download data for {emiten_name}: {e}")

    return RedirectResponse(url=f"/api/ichimoku_data/{emiten_name}")

@exception_handler
@app.get("/api/ichimoku_data_next/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def update_ichimoku_data_next_page(emiten_name: str, current_page: int):
    current_page = current_page + 1
    return RedirectResponse(url=f"/api/ichimoku_data/{emiten_name}/{current_page}")

@exception_handler
@app.get("/api/ichimoku_data_prev/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def update_ichimoku_data_prev_page(emiten_name: str, current_page: int) -> RedirectResponse:
    return RedirectResponse(url=f"/api/ichimoku_data/{emiten_name}/{max(1, current_page - 1)}")

@exception_handler
@app.get("/api/ichimoku_data/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def update_ichimoku_data(emiten_name: str, current_page: int = 1, page_size: int = 10) -> List[Any]:
    ichimoku_data = get_table_data(emiten_name, 'tb_data_ichimoku_cloud')
    ichimoku_data = [IchimokuData(**{**item, 'kode_emiten': emiten_name}) for item in ichimoku_data]

    for item in ichimoku_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    for item in ichimoku_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')

    earliest_date = min(datetime.strptime(item.date, '%d-%m-%Y') for item in ichimoku_data)
    latest_date = max(datetime.strptime(item.date, '%d-%m-%Y') for item in ichimoku_data)

    # if latest_date.date() < datetime.now().date():
    #     try:
    #         data_ic, sen_status, span_status = ichimoku_sql()
    #         if data_ic.empty:
    #             print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
    #         else:
    #             data_ic = pd.DataFrame(data_ic)
    #             stock_id = get_emiten_id(emiten_name)
    #             data_ic['id_emiten'] = stock_id
    #             insert_data_analyst('tb_data_ichimoku_cloud', data_ic)

    #     except Exception as e:
    #         print(f"Failed to download data for {emiten_name}: {e}")

    total_pages = len(ichimoku_data) // page_size
    if len(ichimoku_data) % page_size > 0:
        total_pages += 1

    # Pagination
    start = (current_page - 1) * page_size
    end = start + page_size
    ichimoku_data = ichimoku_data[start:end]

    components_with_next = [
        c.Heading(text=f'Ichimoku Data {emiten_name}', level=2),
        c.Heading(text=f'From {earliest_date.strftime('%Y-%m-%d')} To {latest_date.strftime('%Y-%m-%d')}', level=6),
        c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
        c.Button(text='Previous', on_click=GoToEvent(url=f'/ichimoku_data_prev/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Button(text='Next', on_click=GoToEvent(url=f'/ichimoku_data_next/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Heading(text=f'Deskripsi : Pada page ini berisi data perhitungan lengkap untuk prediksi teknikal Ichimoku Cloud', level=6),
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
        c.Heading(text=f'Page {current_page}', level=6),
        c.Button(text='All Data (No Pagination)', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
    ]

    components_without_next = [
        c.Heading(text=f'Ichimoku Data {emiten_name}', level=2),
        c.Heading(text=f'From {earliest_date.strftime('%Y-%m-%d')} To {latest_date.strftime('%Y-%m-%d')}', level=6),
        c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
        c.Button(text='Previous', on_click=GoToEvent(url=f'/ichimoku_data_prev/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
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
        c.Heading(text=f'Page {current_page}', level=6),
        c.Button(text='All Data (No Pagination)', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
    ]

    return [
        c.Page(
            components=components_with_next if current_page < total_pages else components_without_next
        ),
    ]

@exception_handler
@app.get("/api/error_metrics/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def error_metrics_table(emiten_name: str) -> List[Any]:
    lstm_data = get_table_data(emiten_name, 'tb_lstm')
    lstm_data = [ErrorMetricsResponse(**{**item, 'kode_emiten': emiten_name}) for item in lstm_data]
    for item in lstm_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    for item in lstm_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')
    
    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in lstm_data)
    latest_date = max(item.date for item in lstm_data)
    return [
        c.Page(
            components=[
                c.Heading(text=f'Error Metrics {emiten_name}', level=2),
                c.Button(text='Update Data', on_click=GoToEvent(url=f'/update_error_metrics/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Heading(text=f'From {earliest_date} To {latest_date}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data hasil akurasi dari prediksi akurasi LSTM yang telah dijalankan ', level=6),
                c.Table(
                    data=lstm_data,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='RMSE'),
                        DisplayLookup(field='MAPE'),
                        DisplayLookup(field='MAE'),
                        DisplayLookup(field='MSE'),
                        DisplayLookup(field='accuracy'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@exception_handler
@app.get("/api/update_error_metrics/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def update_error_metrics(emiten_name: str) -> List[Any]:
    lstm_data = get_table_data(emiten_name, 'tb_lstm')
    lstm_data = [ErrorMetricsResponse(**{**item, 'kode_emiten': emiten_name}) for item in lstm_data]
    
    for item in lstm_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')
            
    for item in lstm_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')
    
    # Fetch the latest date    
    latest_date = max(item.date for item in lstm_data)
    
    if latest_date.date() < datetime.now().date():
        # Download the data from Yahoo Finance for the period between the latest date and the current date
        try:
            historical_data = yf.download(emiten_name)
            if historical_data.empty:
                print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
            else:
                model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape, valid, accuracy, gap = train_and_evaluate_model(historical_data, emiten_name)

                historical_df = historical_data.reset_index()
                print(historical_df.tail())

                # id for fk in insert
                stock_id = get_emiten_id(emiten_name)

                # Save data to table 'tb_detail_emiten'
                data_lstm = {
                    'id_emiten': stock_id,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'MAE': mae,
                    'MSE': mse,
                    'accuracy' : accuracy,
                    'date': datetime.now().strftime('%d-%m-%Y')
                }
                insert_data_analyst("tb_lstm", data_lstm)
                
        except Exception as e:
            print(f"Failed to download data for {emiten_name}: {e}")

    return RedirectResponse(url=f"/api/error_metrics/{emiten_name}")

@exception_handler
@app.get("/api/prediction/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def prediction (emiten_name: str) -> List[Any]:
    tb_prediction_lstm = get_table_data(emiten_name, 'tb_prediction_lstm')
    image_path_prediction = f"/static2/prediction/{emiten_name}.png"
    tb_prediction_lstm = [PredictionLSTM(**{**item, 'kode_emiten': emiten_name}) for item in tb_prediction_lstm]
    for item in tb_prediction_lstm:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
            
    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in tb_prediction_lstm)
    latest_date = max(item.date for item in tb_prediction_lstm)
    
    for item in tb_prediction_lstm:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')
            
    return [
        c.Page(
            components=[
                c.Heading(text= f'Error Metrics {emiten_name}', level=2),
                c.Button(text='update data', on_click=GoToEvent(url=f'/update_prediction/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Heading(text=f'From {earliest_date} To {latest_date}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data hasil prediksi harga tertinggi dan terendah beserta tanggal perkiraan terjadinya dengan LSTM', level=6),
                c.Table(
                    data=tb_prediction_lstm,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='max_price'),
                        DisplayLookup(field='max_price_date'),
                        DisplayLookup(field='min_price'),
                        DisplayLookup(field='min_price_date'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
        c.Page(
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
    ]
    
@exception_handler
@app.get("/api/update_prediction/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def update_prediction(emiten_name: str) -> List[Any]:
    tb_prediction_lstm = get_table_data(emiten_name, 'tb_prediction_lstm')
    tb_prediction_lstm = [PredictionLSTM(**{**item, 'kode_emiten': emiten_name}) for item in tb_prediction_lstm]
    for item in tb_prediction_lstm:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    for item in tb_prediction_lstm:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')
    
    # Fetch the latest date
    latest_date = max(item.date for item in tb_prediction_lstm)

    if latest_date < datetime.now().date():
        # Download the data from Yahoo Finance for the period between the latest date and the current date
        try:
            historical_data = yf.download(emiten_name)
            if historical_data.empty:
                print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
            else:
                model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape, valid, accuracy, gap = train_and_evaluate_model(historical_data, emiten_name)
                # Setting up for future predictions
                future_prediction_period = int(len(scaled_data) * 0.1)
                max_price, min_price, max_price_date, min_price_date = predict_future(model, scaler, scaled_data, future_prediction_period, emiten_name)

                historical_df = historical_data.reset_index()
                print(historical_df.tail())

                # id for fk in insert
                stock_id = get_emiten_id(emiten_name)

                # Save data to table 'tb_detail_emiten'
                # Save data to table 'tb_prediction_lstm'
                data_prediction_lstm = {
                    'id_emiten': stock_id,
                    'max_price': max_price,
                    'min_price': min_price,
                    'max_price_date': max_price_date.strftime('%d-%m-%Y'),
                    'min_price_date': min_price_date.strftime('%d-%m-%Y'),
                    'date': datetime.now().strftime('%d-%m-%Y')
                }
                insert_data_analyst('tb_prediction_lstm', data_prediction_lstm)
                
        except Exception as e:
            print(f"Failed to download data for {emiten_name}: {e}")

    return RedirectResponse(url=f"/api/prediction/{emiten_name}")

@exception_handler
@app.get("/api/ichimoku_status/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def ichimoku_status_table(emiten_name: str) -> List[Any]:
    ichimoku_status_data = get_table_data(emiten_name, 'tb_ichimoku_status')
    ichimoku_status_data = [IchimokuStatus(**{**item, 'kode_emiten': emiten_name}) for item in ichimoku_status_data]
    for item in ichimoku_status_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
    
    for item in ichimoku_status_data:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')
            
    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in ichimoku_status_data)
    latest_date = max(item.date for item in ichimoku_status_data)
    return [
        c.Page(
            components=[
                c.Heading(text=f'Ichimoku Status {emiten_name}', level=2),
                c.Heading(text=f'From {earliest_date} To {latest_date}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Button(text='Update Data', on_click=GoToEvent(url=f'/update_lstm_accuracy_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data hasil status prediksi dari perhitungan Ichimoku Cloud', level=6),
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

@exception_handler
@app.get("/api/update_ichimoku_status/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def update_ichimoku_status(emiten_name: str) -> List[Any]:
    ichimoku_status_data = get_table_data(emiten_name, 'tb_ichimoku_status')
    ichimoku_status_data = [IchimokuStatus(**{**item, 'kode_emiten': emiten_name}) for item in ichimoku_status_data]
    for item in ichimoku_status_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
    
    for item in ichimoku_status_data:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')
            
    # Fetch the latest date
    earliest_date = min(item.date for item in ichimoku_status_data)
    latest_date = max(item.date for item in ichimoku_status_data)
    if latest_date.date() < datetime.now().date():
        # Download the data from Yahoo Finance for the period between the latest date and the current date
        try:
            data_ic, sen_status, span_status = ichimoku_sql()
            if data_ic.empty:
                print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
            else:
                stock_id = get_emiten_id(emiten_name)
                data_ic_status = {
                    'id_emiten': stock_id,
                    'sen_status': sen_status,
                    'span_status': span_status,
                    'date': datetime.now().strftime('%d-%m-%Y')
                }
                insert_data_analyst('tb_ichimoku_status', data_ic_status)

        except Exception as e:
            print(f"Failed to download data for {emiten_name}: {e}")

    return RedirectResponse(url=f"/api/ichimoku_status/{emiten_name}")
    
@app.get("/image_list/{path}")
async def get_image_list(path: str):
    image_list = []
    for filename in os.listdir(path):
        if filename.endswith((".png")):
            file_path = os.path.join(path, filename)
            image_url = f"https://lstm-ic.inovasi-digital.my.id/{file_path}"
            image_list.append(image_url)

    return {"image_list": path}

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
    for item in emiten_chart:
        if isinstance(item.render_date, str):
            item.render_date = datetime.strptime(item.render_date, '%Y-%m-%d')
    for item in emiten_chart:
        if isinstance(item.render_date, datetime):
            item.render_date = item.render_date.strftime('%d-%m-%Y')
    return [
        c.Page(
            components=[
                c.Heading(text=f'Charts {emiten_name}', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data gambar summary dari visualisasi data penting dari tiap page', level=6),
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
    for item in accuracy_ichimoku_cloud:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    for item in accuracy_ichimoku_cloud:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')
    
    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in accuracy_ichimoku_cloud)
    latest_date = max(item.date for item in accuracy_ichimoku_cloud)
    return [
        c.Page(
            components=[
                c.Heading(text=f'Ichimoku Accuracy {emiten_name}', level=2),
                c.Heading(text=f'From {earliest_date} To {latest_date}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Button(text='Update Data', on_click=GoToEvent(url=f'/update_ichimoku_accuracy/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data hasil Akurasi dari prediksi teknikal Ichimoku Cloud', level=6),
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
@app.get("/api/update_ichimoku_accuracy/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def update_ichimoku_accuracy(emiten_name: str) -> List[Any]:
    accuracy_ichimoku_cloud = get_table_data(emiten_name, 'tb_accuracy_ichimoku_cloud')
    accuracy_ichimoku_cloud = [IchimokuAccuracy(**{**item, 'kode_emiten': emiten_name}) for item in accuracy_ichimoku_cloud]
    for item in accuracy_ichimoku_cloud:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    for item in accuracy_ichimoku_cloud:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')
            
    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in accuracy_ichimoku_cloud)
    latest_date = max(item.date for item in accuracy_ichimoku_cloud)
    
    if latest_date < datetime.now().date():
        try:
            # Hasil pembuktian
            tren_1hari_sen, tren_1minggu_sen, tren_1bulan_sen = pembuktian_ichimoku(emiten_name, 'sen')
            tren_1hari_span, tren_1minggu_span, tren_1bulan_span = pembuktian_ichimoku(emiten_name, 'span')
            
            # Cek jika list kosong dengan cara yang benar
            if not tren_1hari_sen and not tren_1minggu_sen and not tren_1bulan_sen and not tren_1hari_span and not tren_1minggu_span and not tren_1bulan_span:
                print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
            else:
                # Hitung akurasi jika tidak kosong
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
                
                stock_id = get_emiten_id(emiten_name)
                
                data_accuracy_ichimoku = {
                    'id_emiten': stock_id,
                    'percent_1_hari_sen': percent_1_hari_sen,
                    'percent_1_minggu_sen': percent_1_minggu_sen,
                    'percent_1_bulan_sen': percent_1_bulan_sen,
                    'percent_1_hari_span': percent_1_hari_span,
                    'percent_1_minggu_span': percent_1_minggu_span,
                    'percent_1_bulan_span': percent_1_bulan_span,
                    'date': datetime.now().strftime('%d-%m-%Y')
                }
                insert_data_analyst('tb_accuracy_ichimoku_cloud', data_accuracy_ichimoku)
        
        except Exception as e:
            print(f"Failed to download data for {emiten_name}: {e}")

    return RedirectResponse(url=f"/api/ichimoku_accuracy/{emiten_name}")

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
        result, plot_path, plot_path_2, valid = train_and_evaluate_model2(emiten_name, start_date, end_date, future_date)
        valid = valid.reset_index()
        data = valid.to_dict('records')
        data = [PredicValid(**{**item, 'kode_emiten': emiten_name}) for item in data]
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
            ]
        ), 
        c.Div(
            components=[
                c.Image(src=plot_path_2, alt='LSTM Plot 2', width=1000, height=500, loading='lazy', referrer_policy='no-referrer', class_name='border rounded'),
            ]
        ), 
        c.Div(
            components=[
                c.Table(
                    data=data,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='Predictions'),
                        DisplayLookup(field='Close'),
                        DisplayLookup(field='Date', mode=DisplayMode.date),
                    ],
                )
            ]
        ), 
    ]
    
@exception_handler
@app.get("/api/lstm_accuracy_data/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def lstm_accuracy_table(emiten_name: str) -> List[Any]:
    accuracy_lstm = get_table_data(emiten_name, 'tb_accuracy_lstm')
    accuracy_lstm = [LSTMAccuracy(**{**{k: v if v is not None else 0.0 for k, v in item.items()}, 'kode_emiten': emiten_name}) for item in accuracy_lstm]
    for item in accuracy_lstm:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
            print(f"tanggal : {item.date}")
            
    for item in accuracy_lstm:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in accuracy_lstm)
    print(f"1000: {earliest_date}")
    latest_date = max(item.date for item in accuracy_lstm)
    print(f"1002: {latest_date}")
    return [
        c.Page(
            components=[
                c.Heading(text=f'LSTM Accuracy {emiten_name}', level=2),
                c.Heading(text=f'From {earliest_date} To {latest_date}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Button(text='Update Data', on_click=GoToEvent(url=f'/update_lstm_accuracy_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data accuracy detail untuk prediksi akurasi tertinggi, terendah dan rata - rata', level=6),
                c.Table(
                    data=accuracy_lstm,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                        DisplayLookup(field='mean_gap'),
                        DisplayLookup(field='highest_gap'),
                        DisplayLookup(field='lowest_gap'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]
    
@exception_handler
@app.get("/api/update_lstm_accuracy_data/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def update_lstm_accuracy_data(emiten_name: str) -> List[Any]:
    accuracy_lstm = get_table_data(emiten_name, 'tb_accuracy_lstm')
    accuracy_lstm = [LSTMAccuracy(**{**{k: v if v is not None else 0.0 for k, v in item.items()}, 'kode_emiten': emiten_name}) for item in accuracy_lstm]
    for item in accuracy_lstm:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format
            print(f"tanggal : {item.date}")
            
    for item in accuracy_lstm:
        if isinstance(item.date, date):
            item.date = item.date.strftime('%d-%m-%Y')

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in accuracy_lstm)
    print(f"1000: {earliest_date}")
    latest_date = max(item.date for item in accuracy_lstm)
    print(f"1002: {latest_date}")
    
    if latest_date < datetime.now().date():
        # Download the data from Yahoo Finance for the period between the latest date and the current date
        try:
            historical_data = yf.download(emiten_name)
            if historical_data.empty:
                print(f"No new data found for {emiten_name}. Data might not be available or the ticker might be delisted.")
            else:
                model, scaler, scaled_data, training_data_len, mae, mse, rmse, mape, valid, accuracy, gap = train_and_evaluate_model(historical_data, emiten_name)
                valid_reset = valid.reset_index()

                historical_df = historical_data.reset_index()
                print(historical_df.tail())

                # id for fk in insert
                stock_id = get_emiten_id(emiten_name)

                # Calculate the accuracy of the price predictions
                highest_gap, lowest_gap, mean_gap = gap
                
                data_accuracy_lstm = {
                    'id_emiten': stock_id,
                    'mean_gap': mean_gap,
                    'highest_gap': highest_gap,
                    'lowest_gap': lowest_gap,
                    'date': datetime.now().strftime('%d-%m-%Y')
                }
                insert_data_analyst('tb_accuracy_lstm', data_accuracy_lstm)
                
        except Exception as e:
            print(f"Failed to download data for {emiten_name}: {e}")

    return RedirectResponse(url=f"/api/lstm_accuracy_data/{emiten_name}")

@exception_handler
@app.get("/api/prediction_price_dump_data/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def prediction_price_dump_data_table(emiten_name: str, current_page: int = 1, page_size: int = 10) -> List[Any]:
    prediction_price_dump_data = get_table_data(emiten_name, 'tb_prediction_price_dump_data')
    prediction_price_dump_data = [PredictionPriceResponse(**{**item, 'kode_emiten': emiten_name}) for item in prediction_price_dump_data]

    # Convert the 'date' field to datetime objects if they are strings
    for item in prediction_price_dump_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in prediction_price_dump_data)
    latest_date = max(item.date for item in prediction_price_dump_data)

    # Convert 'date' fields back to strings if needed before returning
    for item in prediction_price_dump_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')
            
    total_pages = len(prediction_price_dump_data) // page_size
    if len(prediction_price_dump_data) % page_size > 0:
        total_pages += 1

    # Pagination
    start = (current_page - 1) * page_size
    end = start + page_size
    prediction_price_dump_data = prediction_price_dump_data[start:end]

    components_with_next = [
        c.Heading(text=f'Prediction Price Dump Data {emiten_name}', level=2),
        c.Heading(text=f'From {earliest_date.strftime("%Y-%m-%d")} To {latest_date.strftime("%Y-%m-%d")}', level=6),
        c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
        c.Button(text='Previous', on_click=GoToEvent(url=f'/prediction_price_dump_data_prev/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Button(text='Next', on_click=GoToEvent(url=f'/prediction_price_dump_data_next/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Table(
            data=prediction_price_dump_data,
            columns=[
                DisplayLookup(field='price'),
                DisplayLookup(field='date', mode=DisplayMode.date),
            ],
        ),
        c.Heading(text=f'Page {current_page}', level=6),
        c.Button(text='Full Data (No Pagination)', on_click=GoToEvent(url=f'/prediction_price_dump_data/{emiten_name}'), named_style='secondary', class_name='ms-2'),
    ]

    components_without_next = [
        c.Heading(text=f'Prediction Price Dump Data {emiten_name}', level=2),
        c.Heading(text=f'From {earliest_date.strftime("%Y-%m-%d")} To {latest_date.strftime("%Y-%m-%d")}', level=6),
        c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
        c.Button(text='Previous', on_click=GoToEvent(url=f'/prediction_price_dump_data_prev/{emiten_name}/{current_page}'), named_style='secondary', class_name='ms-2'),
        c.Table(
            data=prediction_price_dump_data,
            columns=[
                DisplayLookup(field='price'),
                DisplayLookup(field='date', mode=DisplayMode.date),
            ],
        ),
        c.Heading(text=f'Page {current_page}', level=6),
    ]

    return [
        c.Page(
            components=components_with_next if current_page < total_pages else components_without_next
        ),
    ]

@exception_handler
@app.get("/api/prediction_price_dump_data_next/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def prediction_price_dump_data_next_page(emiten_name: str, current_page: int):
    current_page = current_page + 1
    return RedirectResponse(url=f"/api/prediction_price_dump_data/{emiten_name}/{current_page}")

@exception_handler
@app.get("/api/prediction_price_dump_data_prev/{emiten_name}/{current_page}", response_model=FastUI, response_model_exclude_none=True)
def prediction_price_dump_data_prev_page(emiten_name: str, current_page: int) -> RedirectResponse:
    return RedirectResponse(url=f"/api/prediction_price_dump_data/{emiten_name}/{max(1, current_page - 1)}")


@exception_handler
@app.get("/api/prediction_price_dump_data/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def detail_emiten_table(emiten_name: str) -> List[Any]:
    prediction_price_dump_data = get_table_data(emiten_name, 'tb_prediction_price_dump_data')
    prediction_price_dump_data = [PredictionPriceResponse(**{**item, 'kode_emiten': emiten_name}) for item in prediction_price_dump_data]

    # Convert the 'date' field to datetime objects if they are strings
    for item in prediction_price_dump_data:
        if isinstance(item.date, str):
            item.date = datetime.strptime(item.date, '%Y-%m-%d')  # adjust the format string as per your date format

    # Fetch the earliest and latest date
    earliest_date = min(item.date for item in prediction_price_dump_data)
    latest_date = max(item.date for item in prediction_price_dump_data)

    # Run This if i want to click update button
    # update_detail_emiten(latest_date, emiten_name, detail_emiten)

    # Convert 'date' fields back to strings if needed before returning
    for item in prediction_price_dump_data:
        if isinstance(item.date, datetime):
            item.date = item.date.strftime('%d-%m-%Y')
            
    return [
        c.Page(
            components=[
                c.Heading(text=f'Prediction Price Dump Data {emiten_name}', level=2),
                c.Heading(text=f'From {earliest_date.strftime("%Y-%m-%d")} To {latest_date.strftime("%Y-%m-%d")}', level=6),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/navigation/{emiten_name}')),
                c.Table(
                    data=prediction_price_dump_data,
                    columns=[
                        DisplayLookup(field='price'),
                        DisplayLookup(field='date', mode=DisplayMode.date),
            ],
        ),
            ]
        ),
    ]
    
@exception_handler
@app.get("/api/recommendation", response_model=FastUI, response_model_exclude_none=True)
def emiten_recommendation():
    data_1, data_2, data_3, data_4 = fetch_emiten_recommendation()
    
    # Convert lists to sets to remove duplicates
    data_1 = list(set(data_1))
    data_2 = list(set(data_2))
    data_3 = list(set(data_3))
    data_4 = list(set(data_4))

    if len(data_1) == 0 : 
        data_1 = ['Kosong']
    if len(data_2) == 0 : 
        data_2 = ['Kosong']
    if len(data_3) == 0 : 
        data_3 = ['Kosong']
    if len(data_4) == 0 : 
        data_4 = ['Kosong']
    elif not data_1 and not data_2 and not data_3 and not data_4 :  # Check if data list is empty
        return [
            c.Page(
                components=[
                    c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
                    c.Heading(text='Emiten Recommendation', level=2),
                    c.Text(text='No recommendations available.'),
                ]
            ),
        ]

    # Convert to list of dictionaries for the Table component
    data_1 = [Recommendation(kode_emiten=kode, date=date.today()) for kode in data_1]
    data_2 = [Recommendation(kode_emiten=kode, date=date.today()) for kode in data_2]
    data_3 = [Recommendation(kode_emiten=kode, date=date.today()) for kode in data_3]
    data_4 = [Recommendation(kode_emiten=kode, date=date.today()) for kode in data_4]
    
    return [
        c.Page(
            components=[
                c.Heading(text='Emiten Recommendation', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=GoToEvent(url=f'/home')),
                c.Heading(text=f'Deskripsi : Pada page ini berisi data rekomendasi saham sesuai rekomendasi dari perhitungan prediksi LSTM dan Ichimoku Cloud\n', level=6),
                c.Heading(text='Grade 1', level=6),
                c.Table(
                    data=data_1,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                    ],
                ),
                c.Heading(text='Recommendation LSTM', level=6),
                c.Table(
                    data=data_2,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                    ],
                ),
                c.Heading(text='Recommendation IChimoku Cloud', level=6),
                c.Table(
                    data=data_3,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                    ],
                ),
                c.Heading(text='Recommendation Return >5%', level=6),
                c.Table(
                    data=data_4,
                    columns=[
                        DisplayLookup(field='kode_emiten'),
                    ],
                ),
            ]
        ),
    ]
    
@exception_handler
@app.get("/api/testing/{angka}", response_model=FastUI, response_model_exclude_none=True)
def testing(angka: int):  # Define angka as a parameter
    return [
        c.Page(
            components=[
                c.Heading(text=f'angka sekarang : {angka}', level=2),
                c.Button(text='tambah angka', on_click=GoToEvent(url=f'/testing2/{angka}'), named_style='secondary', class_name='ms-2'),
            ]
        ),
    ]
    
@exception_handler
@app.get("/api/testing2/{angka}", response_model=FastUI, response_model_exclude_none=True)
def testing2(angka: int):  # Define angka as a parameter
    angka = angka + angka
    return RedirectResponse(url=f"/api/testing/{angka}")  # Use f-string to format the URL
    
@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))
