import sys 

from src.exception import MyException
from src.logging import logging 
from src.components.data_ingestion import DataIngestion 
from src.components.data_validation import DataValidation 
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.entity.artifacts_entity import DataIngestionArtifact , DataValidationArtifact



class TrainPipeline : 
    
    def __init__(self) : 
        self.data_ingestion_config = DataIngestionConfig()
        self.data_vaidation_config = DataValidationConfig()
        # self.data_ingestion_artifact = DataIngestionArtifact()

    def start_data_ingestion(self)-> DataIngestionArtifact : 
        try: 
            logging.info("Entered the data ingestion  in Train pipeline")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artfact = data_ingestion.initiate_data_ingestion()
            logging.info("got all the data sets from mongodb")

            return data_ingestion_artfact
        except Exception as e : 
            raise MyException(e,sys) 
        
    def start_data_validation(self,data_ingestion_artifact :DataIngestionArtifact) ->DataValidationArtifact: 
        try : 
            logging.info("Entered the data validation in Train Pipeline")

            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=self.data_vaidation_config)

            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data has been validated ")
            return data_validation_artifact
        except Exception as e : 
            raise MyException(e,sys) from e 


    def run_pipeline(self)->None : 
        try: 
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
        except Exception as e : 
            raise MyException(e,sys)

        
