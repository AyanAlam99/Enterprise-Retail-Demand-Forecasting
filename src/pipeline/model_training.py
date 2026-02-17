import sys 

from src.exception import MyException
from src.logging import logging 
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifact



class TrainPipeline : 
    
    def __init__(self) : 
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self)-> DataIngestionArtifact : 
        try: 
            logging.info("Entered the data ingestion  in Train pipeline")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artfact = data_ingestion.initiate_data_ingestion()
            logging.info("got the train and test set from mongodb")
            return data_ingestion_artfact
        except Exception as e : 
            raise MyException(e,sys) 

    def run_pipeline(self)->None : 
        try: 
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e : 
            raise MyException(e,sys)

        
