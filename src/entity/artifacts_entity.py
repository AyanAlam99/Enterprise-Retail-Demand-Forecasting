from dataclasses import dataclass 

@dataclass 
class DataIngestionArtifact : 
    train_file_path: str
    test_file_path: str
    oil_file_path: str
    stores_file_path: str
    holiday_file_path: str
    transactions_file_path: str

@dataclass
class DataValidationArtifact : 
    validation_status : bool 
    message : str 
    validation_report_file_path : str 

@dataclass 
class DataTransformationArtifact: 
    transformed_object_file_path : str 
    transformed_train_file_path :str 
    transformed_test_file_path : str   

@dataclass 
class RegressionMetricArtifact : 
    rmse : float 
    mae : float 
    r2_score : float 


@dataclass 
class ModelTrainerArtifact: 
    trained_model_file_path : str 
    train_metric_artifact : RegressionMetricArtifact
    test_metric_artifact : RegressionMetricArtifact

@dataclass 
class ModelEvaluationArtifact :
    is_model_accepted : bool 
    changed_score : float 
    s3_model_path : str 
    trained_model_path : str 

@dataclass 
class ModelPusherArtifact : 
    bucket_name : str 
    s3_model_path :str