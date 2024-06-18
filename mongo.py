from pymongo import MongoClient
from dotenv import dotenv_values

config = dotenv_values(".env")

def get_database():
    client = MongoClient(config['MONGO_CONNECTION_STRING'])
    database = client['all_ichimoku_data']
    return client 

def shutdown_database():
	client = MongoClient(config['MONGO_CONNECTION_STRING']).close()
	return client

# Function to insert one document into a collection
# params database: the name of the database for example client['all_ichimoku_data'] 
# params collection: the name of the collection for example database['ichimoku_data'] 
# params data: the value of the database for example { } 
def insert_one(database, collection, data):
    collection = database
    collection.insert_one(data) 
    createdValue = collection.find_one(data)
    return createdValue 