from sqlalchemy import create_engine, text

# Create the engine
engine = create_engine('mysql+pymysql://mahaputra971:mahaputra971@localhost:3306/technical_stock_ta_db')

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
            query = text("SELECT kode_emiten FROM tb_emiten limit 2")
            result = connection.execute(query)
            connection.commit()
            for row in result:
                data.append(row[0])
            print("successfully get the data issuer!")
            return data
    except Exception as e:
        print("failed get the data issuer, because:", str(e))


def insert_tables():
    try:
        with engine.connect() as connection:
            # Perform the insert operation here
            # ...
            connection.commit()

            print("Tables inserted successfully!")
    except Exception as e:
        print("Insert failed:", str(e))
