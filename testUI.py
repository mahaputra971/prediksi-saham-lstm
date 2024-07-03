from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastui import FastUI, AnyComponent, prebuilt_html, components as c
from fastui.components.display import DisplayMode, DisplayLookup
from fastui.events import GoToEvent, BackEvent
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, select
from sql import get_table_data

engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')

app = FastAPI()

class TableData(BaseModel):
    table_name: str
    emiten_name: str

@app.post("/api/stock/show/{table_name}/{emiten_name}")
async def show_stock(table_name: str, emiten_name: str):
    try:
        response = get_table_data(emiten_name, table_name)
        if response is None:
            raise HTTPException(status_code=404, detail="Data not available")
        
        df = pd.DataFrame(response)
        if df.empty:
            raise HTTPException(status_code=404, detail="Data not available")

        columns = [DisplayLookup(field=col) for col in df.columns]
        table_data = df.to_dict(orient='records')

        return [
            c.Page(
                components=[
                    c.Heading(text=f'Data for {emiten_name} in {table_name}', level=2),
                    c.Table(
                        data=table_data,
                        columns=columns,
                    ),
                ]
            ),
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def get_table_data_form() -> list[AnyComponent]:
    return [
        c.Page(
            components=[
                c.Heading(text='Enter Table Name and Emiten Name', level=2),
                c.Form(
                    submit_url='/api/stock/show/{table_name}/{emiten_name}',
                    form_fields=[
                        c.FormFieldInput(name='table_name', title='Table Name', type='FormFieldInput'),
                        c.FormFieldInput(name='emiten_name', title='Emiten Name', type='FormFieldInput'),
                    ]
                ),
            ]
        ),
    ]

@app.get('/{path:path}')
async def html_landing() -> HTMLResponse:
    """Simple HTML page which serves the React app, comes last as it matches all paths."""
    return HTMLResponse(prebuilt_html(title='FastUI Demo'))

# def get_table_data(emiten_name: str, table_name: str):
#     metadata = MetaData()
#     table = Table(table_name, metadata, autoload_with=engine)
#     query = select(table).where(table.columns.id_emiten == emiten_name)
#     result = engine.execute(query)
#     data = [dict(row) for row in result]
#     return data if data else None