import pandas as pd 
import pymongo 
from dotenv import load_dotenv
import os
from src.exception import MyException 
import sys 

load_dotenv()
class MongoDB_Connection :

    def __init__(self,DB_NAME : str) : 
        try : 
            self.MONGO_URL = os.getenv("MONGO_URL")
            if self.MONGO_URL is None : 
                raise ValueError("MONGO_URL is not set in .env_file")   
            self.DB_NAME = DB_NAME
            self.client = pymongo.MongoClient(self.MONGO_URL)
            self.data_base = self.client[DB_NAME]
        except Exception as e : 
            raise MyException(e,sys)


    
    def create(self,data_path:str, COLLECTION_NAME:str) :
        try :  
            if not os.path.exists(data_path) : 
                raise FileNotFoundError(f"No file found at {data_path}")
            df = pd.read_csv(data_path)
            data = df.to_dict(orient='records')
            collection = self.data_base[COLLECTION_NAME]
            collection.insert_many(data)
            print(f"{collection} is created at MONGODB")
        except Exception as e : 
            raise MyException(e,sys)

