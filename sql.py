from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from io import BytesIO
from PIL.PngImagePlugin import PngImageFile
import io
import sqlite3
from PIL import Image
import pandas as pd
import datetime

# Create the engine
engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')
Session = sessionmaker(bind=engine)
session = Session()

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
        
def get_issuer():
    data = []
    try:
        with engine.connect() as connection:
            query = text("SELECT kode_emiten FROM tb_emiten limit 1")
            result = connection.execute(query)
            connection.commit()
            for row in result:
                data.append(row[0])
            print("successfully get the data issuer!")
            return data
    except Exception as e:
        print("failed get the data issuer, because:", str(e))

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
                if isinstance(value, datetime.datetime):
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

def convert_image_to_blob(image):
    # Convert image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # Convert bytes to BLOB
    image_blob = sqlite3.Binary(image_bytes)

    return image_blob

def insert_tables():
    try:
        with engine.connect() as connection:
            # Perform the insert operation here
            # ...
            connection.commit()

            print("Tables inserted successfully!")
    except Exception as e:
        print("Insert failed:", str(e))

def truncate_tables(table_name):
    try:
        with engine.connect() as connection:
            if table_name == 'all':
                # Get all table names
                tables = connection.execute(text("SHOW TABLES"))
                for table in tables:
                    # Truncate each table
                    connection.execute(text(f"TRUNCATE TABLE {table[0]}"))
                    print(f"Table {table[0]} truncated successfully.")
            elif table_name == '': 
                ("PLEASE INPUT THE PARAMETER !!!")
            else:
                # Truncate a specific table
                connection.execute(text(f"TRUNCATE TABLE {table_name}"))
                print(f"Table {table_name} truncated successfully.")
    except Exception as e:
        print("An error occurred:", e)