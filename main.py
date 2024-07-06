from pydantic import BaseModel, Field
from typing import List, Any
from datetime import date
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent, BackEvent
from . import get_table_data, exception_handler

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

# Sample data
# detail_emiten = [
#     StockPriceResponse(
#         kode_emiten='ABC',
#         open=100.0,
#         high=110.0,
#         low=90.0,
#         close=105.0,
#         close_adj=103.0,
#         volume=1000,
#         date='2022-01-01'
#     ),
#     StockPriceResponse(
#         kode_emiten='DEF',
#         open=200.0,
#         high=210.0,
#         low=190.0,
#         close=205.0,
#         close_adj=203.0,
#         volume=2000,
#         date='2022-01-02'
#     ),
#     StockPriceResponse(
#         kode_emiten='GHI',
#         open=300.0,
#         high=310.0,
#         low=290.0,
#         close=305.0,
#         close_adj=303.0,
#         volume=3000,
#         date='2022-01-03'
#     ),
#     StockPriceResponse(
#         kode_emiten='JKL',
#         open=400.0,
#         high=410.0,
#         low=390.0,
#         close=405.0,
#         close_adj=403.0,
#         volume=4000,
#         date='2022-01-04'
#     ),
# ]

# ichimoku_data = [
#     IchimokuData(
#         kode_emiten='ABC',
#         close_price=105.0,
#         tenkan_sen=100.0,
#         kijun_sen=95.0,
#         senkou_span_a=110.0,
#         senkou_span_b=90.0,
#         date='2022-01-01'
#     ),
#     IchimokuData(
#         kode_emiten='DEF',
#         close_price=205.0,
#         tenkan_sen=200.0,
#         kijun_sen=195.0,
#         senkou_span_a=210.0,
#         senkou_span_b=190.0,
#         date='2022-01-02'
#     ),    
#     IchimokuData(
#         kode_emiten='GHI',
#         close_price=305.0,
#         tenkan_sen=300.0,
#         kijun_sen=295.0,
#         senkou_span_a=310.0,
#         senkou_span_b=290.0,
#         date='2022-01-03'
#     ),
#     IchimokuData(
#         kode_emiten='JKL',
#         close_price=405.0,
#         tenkan_sen=400.0,
#         kijun_sen=395.0,
#         senkou_span_a=410.0,
#         senkou_span_b=390.0,
#         date='2022-01-04'
#     ),
# ]

# error_metrics_response = [
#     ErrorMetricsResponse(
#         kode_emiten='ABC',
#         RMSE=0.1,
#         MAPE=0.05,
#         MAE=0.02,
#         MSE=0.01,
#         date='2022-01-01'
#     ),
#     ErrorMetricsResponse(
#         kode_emiten='DEF',
#         RMSE=0.2,
#         MAPE=0.1,
#         MAE=0.04,
#         MSE=0.02,
#         date='2022-01-02'
#     ),
#     ErrorMetricsResponse(
#         kode_emiten='GHI',
#         RMSE=0.3,
#         MAPE=0.15,
#         MAE=0.06,
#         MSE=0.03,
#         date='2022-01-03'
#     ),
#     ErrorMetricsResponse(
#         kode_emiten='JKL',
#         RMSE=0.4,
#         MAPE=0.2,
#         MAE=0.08,
#         MSE=0.04,
#         date='2022-01-04'
#     ),
# ]

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

# Fetch data from database
# detail_emiten = get_table_data('BELI.JK', 'tb_detail_emiten')
# ichimoku_data = get_table_data('BELI.JK', 'tb_data_ichimoku_cloud')
# error_metrics_response = get_table_data('BELI.JK', 'tb_prediction_lstm')
# chart_response = get_table_data('BELI.JK', 'tb_summary')

# Convert fetched data to response models
# detail_emiten = [StockPriceResponse(**item) for item in detail_emiten]
# ichimoku_data = [IchimokuData(**item) for item in ichimoku_data]
# error_metrics_response = [ErrorMetricsResponse(**item) for item in error_metrics_response]
# chart_response = [ChartResponse(**item) for item in chart_response]


@exception_handler
@app.get("/api/home", response_model=FastUI, response_model_exclude_none=True)
def home() -> List[Any]:
    return [
        c.Page(
            components=[
                c.Heading(text='Navbar', level=2),
                c.Button(text='Detail Emiten', on_click=GoToEvent(url='/detail_emiten/')),
                c.Button(text='Ichimoku Data', on_click=GoToEvent(url='/ichimoku_data/')),
                c.Button(text='Error Metrics', on_click=GoToEvent(url='/error_metrics/')),
                c.Button(text='Charts', on_click=GoToEvent(url='/charts/')),
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

@app.get("/api/charts/", response_model=FastUI, response_model_exclude_none=True)
def charts_table() -> List[Any]:
    return [
        c.Page(
            components=[
                c.Heading(text='Charts', level=2),
                c.Link(components=[c.Text(text='Back')], on_click=BackEvent()),
                c.Table(
                    data=chart_response,
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


