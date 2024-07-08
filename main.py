from pydantic import BaseModel, Field
from typing import List, Any
from datetime import date
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent, BackEvent
from fastui.forms import fastui_form
from . import get_table_data, exception_handler
from fastapi.responses import RedirectResponse

class EmitenForm(BaseModel):
    emiten_name: str = Field(title="Emiten Name")

app = FastAPI()

# Define models
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

chart_response = [
    ChartResponse(
        kode_emiten='ABC',
        pic_closing_price='pic_closing_price_abc.png',
        pic_sales_volume='pic_sales_volume_abc.png',
        pic_price_history='pic_price_history_abc.png',
        pic_comparation='pic_comparation_abc.png',
        pic_prediction='pic_prediction_abc.png',
        pic_ichimoku_cloud='pic_ichimoku_cloud_abc.png',
        render_date='2022-01-01'
    ),
    ChartResponse(
        kode_emiten='DEF',
        pic_closing_price='pic_closing_price_def.png',
        pic_sales_volume='pic_sales_volume_def.png',
        pic_price_history='pic_price_history_def.png',
        pic_comparation='pic_comparation_def.png',
        pic_prediction='pic_prediction_def.png',
        pic_ichimoku_cloud='pic_ichimoku_cloud_def.png',
        render_date='2022-01-02'
    ),
    ChartResponse(
        kode_emiten='GHI',
        pic_closing_price='pic_closing_price_ghi.png',
        pic_sales_volume='pic_sales_volume_ghi.png',
        pic_price_history='pic_price_history_ghi.png',
        pic_comparation='pic_comparation_ghi.png',
        pic_prediction='pic_prediction_ghi.png',
        pic_ichimoku_cloud='pic_ichimoku_cloud_ghi.png',
        render_date='2022-01-03'
    ),
    ChartResponse(
        kode_emiten='JKL',
        pic_closing_price='pic_closing_price_jkl.png',
        pic_sales_volume='pic_sales_volume_jkl.png',
        pic_price_history='pic_price_history_jkl.png',
        pic_comparation='pic_comparation_jkl.png',
        pic_prediction='pic_prediction_jkl.png',
        pic_ichimoku_cloud='pic_ichimoku_cloud_jkl.png',
        render_date='2022-01-04'
    ),
]

@app.get("/api/home", response_model=FastUI, response_model_exclude_none=True)
async def home() -> list[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Heading(text='Navbar', level=2),
                c.ModelForm(model=EmitenForm, display_mode='page', submit_url='/api/submit_emiten_form'),
            ]
        ),
    ]

@app.post("/api/submit_emiten_form", response_model=FastUI, response_model_exclude_none=True)
async def submit_emiten_form(emiten_name: str = Form(...)):
    return [
        c.Page(
            components=[
                c.Heading(text='Select Action for Emiten', level=2),
                c.Button(text='Detail Emiten', on_click=GoToEvent(url=f'/detail_emiten/{emiten_name}')),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url=f'/ichimoku_data/{emiten_name}')),
                c.Button(text='Error Metrics', on_click=GoToEvent(url=f'/error_metrics/{emiten_name}')),
                c.Button(text='Charts', on_click=GoToEvent(url=f'/charts/{emiten_name}')),
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
                c.Link(components=[c.Text(text='Back')], on_click=BackEvent()),
                c.Table(
                    data=detail_emiten,
                    columns=[
                        DisplayLookup(field='kode_emiten', on_click=GoToEvent(url='/emiten/{kode_emiten}/')),
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
                c.Link(components=[c.Text(text='Back')], on_click=BackEvent()),
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
                c.Link(components=[c.Text(text='Back')], on_click=BackEvent()),
                c.Table(
                    data=lstm_data,
                    columns=[
                        DisplayLookup(field='kode_emiten', on_click=GoToEvent(url='/emiten/{kode_emiten}/error_metrics')),
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

@app.get("/api/charts/{emiten_name}", response_model=FastUI, response_model_exclude_none=True)
def charts_table(emiten_name: str) -> List[Any]:
    emiten_charts = [chart for chart in chart_response if chart.kode_emiten == emiten_name]
    return [
        c.Page(
            components=[
                c.Heading(text='Charts', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=BackEvent()),
                c.Table(
                    data=emiten_charts,
                    columns=[
                        DisplayLookup(field='kode_emiten', on_click=GoToEvent(url='/emiten/{kode_emiten}/charts')),
                        DisplayLookup(field='pic_closing_price', mode=DisplayMode.image),
                        DisplayLookup(field='pic_sales_volume', mode=DisplayMode.image),
                        DisplayLookup(field='pic_price_history', mode=DisplayMode.image),
                        DisplayLookup(field='pic_comparation', mode=DisplayMode.image),
                        DisplayLookup(field='pic_prediction', mode=DisplayMode.image),
                        DisplayLookup(field='pic_ichimoku_cloud', mode=DisplayMode.image),
                        DisplayLookup(field='render_date', mode=DisplayMode.date),
                    ],
                ),
            ]
        ),
    ]

@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))
