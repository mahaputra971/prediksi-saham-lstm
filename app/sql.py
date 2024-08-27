# app/sql.py

from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select
from io import BytesIO
import io
import sqlite3
from PIL import Image
import pandas as pd
from datetime import datetime, date, timedelta
from tensorflow.keras.models import load_model, save_model
import tempfile
from app.exception import exception_handler

from dotenv import load_dotenv
import os

load_dotenv()  # take environment variables

# Create the engine
engine = create_engine(os.getenv('MYSQL_STRING'))
Session = sessionmaker(bind=engine)
session = Session()

@exception_handler
def get_emiten_status(emiten_name):
    print(emiten_name)
    try:
        result = session.execute(text(f"SELECT status FROM tb_emiten WHERE kode_emiten = :emiten_name"), {'emiten_name': emiten_name})
        status = result.scalar()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        status = None
    finally:
        session.close()
    return status

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
def get_emiten_id(stock_value):
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
            query = text("SELECT kode_emiten, nama_emiten FROM tb_emiten WHERE status = 0")
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
        id_emiten = get_emiten_id(emiten_name)
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

@exception_handler
def save_model_to_db(model, id_emiten, name, algorithm, hyperparameters, metrics):
    try:
        # Serialize the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
            model.save(tmp.name)
            tmp.seek(0)
            model_blob = tmp.read()

        # Create the data dictionary
        data = {
            'id_emiten': id_emiten,
            'name': name,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_blob': model_blob,
            'algorithm': algorithm,
            'hyperparameters': str(hyperparameters),
            'metrics': str(metrics)
        }

        # Insert the model into the database
        insert_data_analyst("models", data)
        print("Model saved to database successfully.")
    except Exception as e:
        print("An error occurred while saving the model:", e)

@exception_handler
def save_model_to_directory(model, id_emiten, name, algorithm, hyperparameters, metrics):
    try:
        # Define the directory and filename
        directory = 'model'
        filename = f'{name}.h5'

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the model to the file
        model.save(os.path.join(directory, filename))

        print("Model saved to directory successfully.")
    except Exception as e:
        print("An error occurred while saving the model:", e)

@exception_handler
def load_model_from_db(model_id):
    try:
        # Fetch the model blob from the database
        query = text("SELECT model_blob FROM models WHERE id_model = :model_id")
        model_blob = session.execute(query, {'model_id': model_id}).scalar()

        if model_blob is None:
            print(f"No model found with ID: {model_id}")
            return None

        # Deserialize the model from the blob
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=True) as tmp:
            tmp.write(model_blob)
            tmp.flush()
            model = load_model(tmp.name)

        print("Model loaded from database successfully.")
        return model
    except Exception as e:
        print("An error occurred while loading the model:", e)
        return None
    
@exception_handler
def load_model_from_directory(model_name):
    try:
        # Define the directory and filename
        directory = 'model'
        filename = f'{model_name}.h5'

        # Check if the file exists
        if not os.path.exists(os.path.join(directory, filename)):
            print(f"No model found with name: {model_name}")
            return None

        # Load the model from the file
        model = load_model(os.path.join(directory, filename))

        print("Model loaded from directory successfully.")
        return model
    except Exception as e:
        print("An error occurred while loading the model:", e)
        return None

@exception_handler
def get_model_id_by_emiten(id_emiten):
    try:
        # Fetch the latest model_id for the given id_emiten
        query = text("SELECT id_model FROM models WHERE id_emiten = :id_emiten ORDER BY created_at DESC LIMIT 1")
        model_id = session.execute(query, {'id_emiten': id_emiten}).scalar()
        return model_id
    except Exception as e:
        print("An error occurred while fetching the model ID:", e)
        return None


import yfinance as yf
from pandas_datareader.data import DataReader
yf.pdr_override()

@exception_handler
def fetch_stock_data(stock_list, start, end):
    try:
        data = {stock: yf.download(stock, start=start, end=end) for stock in stock_list}
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None
    
@exception_handler
def fetch_emiten_recommendation():
    try:
        with engine.connect() as connection:
            query_grade_1 = text("""
                SELECT tb_emiten.kode_emiten
                FROM tb_emiten
                JOIN tb_lstm ON tb_emiten.id_emiten = tb_lstm.id_emiten
                JOIN tb_ichimoku_status ON tb_emiten.id_emiten = tb_ichimoku_status.id_emiten
                WHERE tb_lstm.accuracy > 80
                AND tb_ichimoku_status.sen_status IN ('Pasar Bullish', 'Pasar Bullish Konsolidasi, Potensi Kelanjutan Kenaikan')
                AND tb_ichimoku_status.span_status IN ('Senkou_Span Uptrend', 'Senkou_Span Will Pump', 'Senkou_Span Uptrend and Will Bounce Up')
            """)
            result_1 = connection.execute(query_grade_1)
            data_1 = [row[0] for row in result_1]
            print(f'The Recommendation : {data_1}')
            
            query_grade_2 = text("""
                SELECT tb_emiten.kode_emiten
                FROM tb_emiten
                JOIN tb_lstm ON tb_emiten.id_emiten = tb_lstm.id_emiten
                JOIN tb_ichimoku_status ON tb_emiten.id_emiten = tb_ichimoku_status.id_emiten
                WHERE tb_lstm.accuracy > 80
            """)
            result_2 = connection.execute(query_grade_2)
            data_2 = [row[0] for row in result_2]
            print(f'The Recommendation : {data_2}')
            
            query_grade_3 = text("""
                SELECT tb_emiten.kode_emiten
                FROM tb_emiten
                JOIN tb_lstm ON tb_emiten.id_emiten = tb_lstm.id_emiten
                JOIN tb_ichimoku_status ON tb_emiten.id_emiten = tb_ichimoku_status.id_emiten
                WHERE tb_ichimoku_status.sen_status IN ('Pasar Bullish', 'Pasar Bullish Konsolidasi, Potensi Kelanjutan Kenaikan')
                AND tb_ichimoku_status.span_status IN ('Senkou_Span Uptrend', 'Senkou_Span Will Pump', 'Senkou_Span Uptrend and Will Bounce Up')
            """)
            result_3 = connection.execute(query_grade_3)
            data_3 = [row[0] for row in result_3]
            print(f'The Recommendation : {data_3}')
            
            return data_1, data_2, data_3
    except Exception as e:
        print("An error occurred:", e)
        return None
