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

DATA_INGESTION_COLLECTION_NAME = ["Train_Data", "Oil", "Stores", "Holidays_Events", "Transactions"]
OOT_SPLIT_DAYS: int = 30

SCHEMA_FILE_PATH = "./config/schema_config.yaml"

DATA_VALIDATION_DIR_NAME : str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME :str = "report.yaml"

DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation" 
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR :str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR :str = "transformed_object"

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"


MODEL_TRAINER_DIR_NAME: str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR :str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME : str = 'model.pkl'
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH : str = os.path.join('config','model.yaml')
MODEL_TRAINER_EXPECTED_SCORE : float = 0.85
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05

MODEL_FILE_NAME :str = 'model.pkl'


AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_ACCESS_KEY"
REGION_NAME : str = 'us-east-1'


MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE : float = 0.02
MODEL_BUCKET_NAME : str = "enterprise-model-proj"
MODEL_PUSHER_S3_KEY : str = "model-registry"
##


