import json 
import os 
import sys 
import pandas as pd 
from pandas import DataFrame

from src.exception  import MyException
from src.logging import logging
from src.utils.main_utils import read_yaml 
from src.entity.artifacts_entity import DataIngestionArtifact , DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import *


class DataValidation: 
    def __init__(self,data_ingestion_artifact : DataIngestionArtifact,data_validation_artifact : DataValidationArtifact) :
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
        except Exception as e : 
           raise  MyException(e,sys) from e
    
    def validate_number_columns(self, df : DataFrame) ->bool: 
        try : 
            status = len(df.columns) == len(self._schema_config["columns"])
            logging.info(f"Is required columns present : {[status]}")
            return status
        except Exception as e : 
            raise MyException(e,sys) from e 
        
    def is_column_exist(self,df : DataFrame) ->bool : 
        try : 
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["columns"] : 
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logging.info(f"Missing categorical column: {missing_categorical_columns}")
        
            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True
        except Exception as e:
            raise MyException(e, sys) from e
        
    
    @staticmethod

    def read_data(file_path : str) -> DataFrame : 
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_validation(self) ->DataValidationArtifact : 
        try : 
            validation_error_mssg = ""
            data_to_validate = {
                "train" : self.data_ingestion_artifact.train_file_path,
                "test" : self.data_ingestion_artifact.test_file_path,
                "oil" : self.data_ingestion_artifact.oil_file_path,
                "holiday" : self.data_ingestion_artifact.holiday_file_path,
                "store" : self.data_ingestion_artifact.stores_file_path,
                "transactions" : self.data_ingestion_artifact.transactions_file_path
            }

            for key , value in data_to_validate.items() : 
                df = DataValidation.read_data(value)
                status_num = self.validate_number_columns(df=df)
                status= self.is_column_exist(df=df)

                if not status or not status_num : 
                    validation_error_mssg += f"Columns are missing in {key} dataframe. "
                else:
                    logging.info(f"All required columns present in {key} dataframe: {status}")

            validation_status = len(validation_error_mssg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_mssg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

      
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)


            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_mssg.strip()
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e





         