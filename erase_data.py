from integrations import truncate_tables
from sqlalchemy import text, create_engine
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from datetime import timedelta, datetime

import os

load_dotenv()  # take environment variables

# Create the engine
engine = create_engine(os.getenv('MYSQL_STRING'))
Session = sessionmaker(bind=engine)
session = Session()
today = datetime.now().strftime("%Y-%m-%d")


# show_specific_tables('tb_accuracy_ichimoku_cloud')
truncate_tables('tb_accuracy_ichimoku_cloud')
truncate_tables('tb_accuracy_lstm')
truncate_tables('tb_data_ichimoku_cloud')
truncate_tables('tb_detail_emiten')
truncate_tables('tb_ichimoku_status')
truncate_tables('tb_lstm')
truncate_tables('tb_prediction_lstm')
truncate_tables('tb_prediction_lstm_data')
truncate_tables('tb_summary')

# Set the 'status' column in 'tb_emiten' to '1' for the given stock
try:
    update_query = text("UPDATE tb_emiten SET status = 0 WHERE status = 1")
    session.execute(update_query)
    session.commit()
    print("Success erase data")
except Exception as e:
    print(f"Commit ERROR: {str(e)}")