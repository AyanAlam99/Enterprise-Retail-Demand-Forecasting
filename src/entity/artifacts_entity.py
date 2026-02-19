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
