import sys 

from src.cloud_storage.aws_storage import SimpleStorageServices
from src.exception import MyException 
from src.logging import logging
from src.entity.config_entity import ModelPusherConfig 
from src.entity.artifacts_entity import ModelEvaluationArtifact , ModelPusherArtifact
from src.entity.s3_estimator import ProjEstimator

class ModelPusher : 
    def __init__(self,model_evaluation_artifact : ModelEvaluationArtifact , model_pusher_config : ModelPusherConfig): 
        self.s3 = SimpleStorageServices()
        self.model_pusher_config= model_pusher_config 
        self.model_evaluation_artifact = model_evaluation_artifact 
        self.proj1_estimator = ProjEstimator(bucket_name=self.model_pusher_config.bucket_name , model_path=self.model_pusher_config.s3_model_key_path)

    
    def initiate_model_pusher(self)->ModelPusherArtifact : 
        logging.info("Entered initiate_model_pusher method of ModelTrainer class ")

        try : 
            logging.info("Uploadinf new model to S3 bucket ")
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name , s3_model_path=self.model_pusher_config.s3_model_key_path)

            logging.info("uploaded artifacts to s3 bucket")
            logging.info("exited initiated model pusher method ")
            return model_pusher_artifact
        except Exception as e : 
            raise MyException(e,sys) from e 
        