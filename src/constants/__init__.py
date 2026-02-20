import os 
from datetime import date 
PIPELINE_NAME:str = ""
ARTIFACT_DIR:str = "artifact"


# DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25 


FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

DATA_INGESTION_COLLECTION_NAME = ["Train_Data", "Test", "Oil", "Stores", "Holidays_Events", "Transactions"]

SCHEMA_FILE_PATH = "./config/schema_config.yaml"

DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME :str = "report.yaml"

DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation" 
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR :str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR :str = "transformed_object"