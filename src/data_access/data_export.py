import sys 
import pandas as pd 
import numpy as np 
from typing import Optional
from dotenv import load_dotenv
import os 

from src.configuration.mongodb_connection import MongoDB_Connection 
from src.exception import MyException

load_dotenv()

class ExportData : 
    """
    Class to export data records from mongodb as pd dataframe
    """
    def __init__(self) ->None : 
        try : 
            self.database_name = os.getenv("DB_NAME")
            self.client = MongoDB_Connection(DB_NAME=self.database_name)
        except Exception as e : 
            raise MyException(e,sys)
    
    def export_collection_as_dataframe(self,collection_name : str,database_name :Optional[str] = None) ->pd.DataFrame : 
        """
        basically if you want to export data from a specific database , then you have to pass that database name as an arg here , otherwise if you wont , it will take the default one that has been used while establishing the mongodb connction at init 
        """
        try : 
            if database_name is None : 
                collection = self.client.data_base[collection_name]
            else : 
                collection = self.client.client[database_name][collection_name]
            print(f'Fetching data from mongodb')
            df = pd.DataFrame(list(collection.find()))
            print(f"Data ftched with len {len(df)}")
            
            if "_id" in df.columns.to_list() : 
                df = df.drop(columns=["_id"],axis=1)
            df.replace({"na" : np.nan},inplace=True)  #as pandas function only consider np.nan as missing value
            return df
        
        except Exception as e : 
            raise MyException(e,sys)