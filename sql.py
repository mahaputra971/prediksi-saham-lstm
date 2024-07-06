from sqlalchemy import create_engine, text, create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from io import BytesIO
from PIL.PngImagePlugin import PngImageFile
import io
import sqlite3
from PIL import Image
import pandas as pd
from datetime import datetime, date
from . import exception_handler

# Create the engine
engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')
Session = sessionmaker(bind=engine)
session = Session()

@exception_handler
def show_tables():
    try:
        with engine.connect() as connection:
            tables = connection.execute(text("SHOW TABLES"))
            connection.commit()

            for row in tables.mappings():
                print("Tables:", row)

            print("Connected successfully!!!!!!!!!!!!!!!!")
    except Exception as e:
        print("Connection failed:", str(e))

@exception_handler
def get_emiten_id(stock_value, table_name):
    try:
        # Get the emiten_id from the tb_emiten table
        emiten_row = session.execute(text(f"SELECT id_emiten FROM tb_emiten WHERE kode_emiten = :stock"), {'stock': stock_value}).first()
        if emiten_row is not None:
            return emiten_row.id_emiten
        else:
            print(f"No record found for stock: {stock_value}")
            return None
    except Exception as e:
        print("An error occurred:", e)
        return None

@exception_handler
def show_specific_tables(table_name):
    data = []
    try:
        with engine.connect() as connection:
            query = text(f"SELECT * FROM {table_name}")
            result = connection.execute(query)
            connection.commit()
            for row in result:
                data.append(row)
                # print(row)
            print("Table displayed successfully!!!!!!!!!!!!!!!!!")
            return data
    except Exception as e:
        print("Displaying table failed:", str(e))
        
@exception_handler
def get_issuer():
    data_code = []
    data_name = []
    try:
        with engine.connect() as connection:
            query = text("SELECT kode_emiten, nama_emiten FROM tb_emiten limit 1")
            result = connection.execute(query)
            for row in result:
                data_code.append(row[0])
                data_name.append(row[1])
            print("successfully get the data issuer!")
            return data_code, data_name
    except Exception as e:
        print("failed get the data issuer, because:", str(e))

@exception_handler
def insert_data_analyst(table_name, data):
    try:
        # If data is a DataFrame, convert it to a list of dictionaries
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient='records')

        # If data is a dictionary, convert it to a list containing one dictionary
        if isinstance(data, dict):
            data = [data]

        for row in data:
            # Convert images in row to BLOBs
            for key, value in row.items():
                # print(f"{key}: {type(value)}") #debug type data
                if isinstance(value, Image.Image):
                    row[key] = convert_image_to_blob(value)
                    
                # Convert datetime.datetime objects to strings
                if isinstance(value, datetime):
                    row[key] = value.strftime('%Y-%m-%d')
                    
                # Convert datetime.date objects to strings
                if isinstance(value, date):
                    row[key] = value.strftime('%Y-%m-%d')

            # Create query
            columns = ', '.join(row.keys())
            placeholders = ', '.join(':' + key for key in row.keys())
            query = text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})")

            # Execute query
            session.execute(query, row)
            session.commit()
            print("Data inserted successfully.")
    except Exception as e:
        session.rollback()
        print("An error occurred:", e)
    finally:
        session.close()

@exception_handler
def convert_image_to_blob(image):
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Convert bytes to BLOB
    image_blob = sqlite3.Binary(image_bytes)

    return image_blob

@exception_handler
def insert_tables():
    try:
        with engine.connect() as connection:
            # Perform the insert operation here
            # ...
            connection.commit()

            print("Tables inserted successfully!")
    except Exception as e:
        print("Insert failed:", str(e))

@exception_handler
def truncate_tables(table_name):
    try:
        with engine.connect() as connection:
            if table_name == 'all':
                # Define the order of tables for truncation
                tables = ['tb_lstm', 'tb_ichimoku_status', 'tb_data_ichimoku_cloud', 'id_detail_emiten', 'id_emiten']
                for table in tables:
                    # Truncate each table
                    connection.execute(text(f"TRUNCATE TABLE {table}"))
                    print(f"Table {table} truncated successfully.")
            elif table_name == '': 
                print("PLEASE INPUT THE PARAMETER !!!")
            else:
                # Truncate a specific table
                connection.execute(text(f"TRUNCATE TABLE {table_name}"))
                print(f"Table {table_name} truncated successfully.")
    except Exception as e:
        print("An error occurred:", e)

@exception_handler
def get_table_data(emiten_name, table_name):
    with engine.connect() as connection:
        id_emiten = get_emiten_id(emiten_name, table_name)
        # Reflect the table from the database
        metadata = MetaData()
        table = Table(table_name, metadata, autoload_with=engine)

        # Query all rows from the table where the id_emiten matches
        query = select(table).where(table.c.id_emiten == id_emiten)
        result = connection.execute(query).fetchall()

        # Map the columns to a list of dictionaries
        response = []
        for row in result:
            item = {}
            for i, column in enumerate(table.columns):
                value = row[i]
                if isinstance(value, datetime):
                    # Convert datetime objects to string
                    item[column.name] = value.isoformat()
                elif isinstance(value, date):
                    # Convert date objects to string
                    item[column.name] = value.isoformat()
                else:
                    item[column.name] = value
            response.append(item)

        # Return the list of dictionaries
        return response

