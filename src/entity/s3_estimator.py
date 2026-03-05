from src.cloud_storage.aws_storage import SimpleStorageServices 
from src.exception import MyException 
from src.logging import logging
from src.entity.estimator import MyModel 
import sys 
from pandas import DataFrame


class ProjEstimator : 

    def __init__(self, bucket_name, model_path) : 
        self.bucket_name = bucket_name 
        self.s3 = SimpleStorageServices()
        self.model_path = model_path 
        self.loaded_model : MyModel = None 

    
    def is_model_present(self, model_path) : 
        try : 
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name ,s3_key=model_path)
        except Exception as e : 
            print(e)
            return False
        
    def load_model(self)->MyModel : 
        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)
    
    def save_model(self, from_file,remove:bool= False) -> None : 
        try : 
            self.s3.upload_file(
                from_filename = from_file, 
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove    
            )
        except Exception as e : 
            raise MyException(e,sys) from e 
        
    def predict(self,dataframe : DataFrame) : 
        try : 
            if self.loaded_model is None : 
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(raw_dataframe=dataframe)
        except Exception as e :
            raise MyException(e,sys) from e
            
        