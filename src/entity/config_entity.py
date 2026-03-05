import os 
from src.constants import * 
from dataclasses import dataclass ,field
from datetime import datetime
from typing import List


TIMESTAMP :str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig : 
    pipeline_name : str = PIPELINE_NAME
    artifact_dir : str = os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp : str = TIMESTAMP

training_pipeline_config : TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig : 
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    raw_data_dir: str = os.path.join(data_ingestion_dir, "raw_data")
    ingested_dir: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)
    training_file_path: str = os.path.join(ingested_dir, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(ingested_dir, TEST_FILE_NAME)
    collection_names : List= field (default_factory = lambda:DATA_INGESTION_COLLECTION_NAME)

@dataclass
class DataValidationConfig : 
    data_validation_dir : str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path : str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)


@dataclass 
class DataTransformationConfig : 
    data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir , DATA_TRANSFORMATION_DIR_NAME)
    transformaion_train_file_path : str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, TRAIN_FILE_NAME.replace('csv','npy'))
    transformed_test_file_path :str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FILE_NAME.replace('csv','npy'))
    transformed_object_file_path : str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCSSING_OBJECT_FILE_NAME)


@dataclass 
class ModelTrainerConfig : 
    model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
    trained_model_file_path = os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_FILE_NAME)
    model_config_file_path = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    expected_score : float = MODEL_TRAINER_EXPECTED_SCORE 
    overfitting_underfitting_threshold: float = MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD

@dataclass 
class ModelEvaluationConfig : 
    changed_threshold_score : float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE 
    bucket_name : str = MODEL_BUCKET_NAME
    s3_model_key_path : str = MODEL_FILE_NAME

@dataclass 
class ModelPusherConfig : 
    bucket_name : str = MODEL_BUCKET_NAME 
    s3_model_key_path : str = MODEL_FILE_NAME   
    

@dataclass 
class PredictionPipelineConfig : 
    bucket_name = MODEL_BUCKET_NAME
    model_file_path: str = MODEL_FILE_NAME