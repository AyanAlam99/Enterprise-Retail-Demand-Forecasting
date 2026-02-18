import os 
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import * 
from src.data_access.data_export import ExportData
from src.exception import MyException
from src.entity.config_entity import DataIngestionConfig 
from src.entity.artifacts_entity import DataIngestionArtifact   
from src.logging import logging

class DataIngestion : 

    def __init__(self,data_ingestion_config : DataIngestionConfig =  DataIngestionConfig()) : 
        try : 
            self.data_ingestion_config = data_ingestion_config
        except Exception as e : 
            MyException(e,sys)

    def export_data_into_feature_store(self)->pd.DataFrame : 
        try : 
            logging.info(f"Exporting data from mongodb")
            mydata  =ExportData()
            collections = self.data_ingestion_config.collection_names
            raw_dir_file_path = self.data_ingestion_config.raw_data_dir
            dir_path = os.path.dirname(raw_dir_file_path)
            os.makedirs(raw_dir_file_path,exist_ok=True)
            for col_name in collections:
                logging.info(f"Exporting collection: {col_name}")
                df = mydata.export_collection_as_dataframe(collection_name=col_name)
                
               
                file_path = os.path.join(self.data_ingestion_config.raw_data_dir, f"{col_name}.csv")
                df.to_csv(file_path, index=False, header=True)
                logging.info(f"Saved {col_name} to {file_path}")

        except Exception as e : 
            raise MyException(e,sys)
        
    def split_data_train_test(self,dataframe:pd.DataFrame):
        try : 
            logging.info("Entered into train test spllit of the dataframe")

            training_df , testing_df = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            train_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            test_dir = os.path.dirname(self.data_ingestion_config.testing_file_path)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            training_df.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            testing_df.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info(f"Exported train and test file path.")
        except Exception as e : 
            raise MyException(e , sys)
        
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        logging.info("Entered initiate data ingestion method in dataingestion")
        try : 
            data = self.export_data_into_feature_store()
            logging.info("Got data from mongodb")
            # self.split_data_train_test(data)
            raw_dir = self.data_ingestion_config.raw_data_dir

            data_ingestion_artifact = DataIngestionArtifact(
                        train_file_path = os.path.join(raw_dir, "Train_Data.csv"),
                        test_file_path = os.path.join(raw_dir, "Test.csv"),
                        oil_file_path = os.path.join(raw_dir, "Oil.csv"),
                        stores_file_path = os.path.join(raw_dir, "Stores.csv"),
                        holiday_file_path = os.path.join(raw_dir, "Holidays_Events.csv"),
                        transactions_file_path = os.path.join(raw_dir, "Transactions.csv"),
                    )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

     
