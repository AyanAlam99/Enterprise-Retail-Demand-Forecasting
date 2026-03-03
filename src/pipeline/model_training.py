import sys 

from src.exception import MyException
from src.logging import logging 
from src.components.data_ingestion import DataIngestion 
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig, ModelTrainerConfig , ModelEvaluationConfig , ModelPusherConfig 
from src.entity.artifacts_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact , ModelTrainerArtifact  , ModelEvaluationArtifact , ModelPusherArtifact



class TrainPipeline : 
    
    def __init__(self) : 
        self.data_ingestion_config = DataIngestionConfig()
        self.data_vaidation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
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

    
    def start_data_transformation(self,data_ingestion_artifact : DataIngestionArtifact  , data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact : 
        try : 
            logging.info("Entered the data transformation in Train Pipeline")

            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact,data_transformation_config=self.data_transformation_config)

            data_transformation_artifact = data_transformation.initiate_data_transformaion()

            logging.info("Data has been transformed")
            return data_transformation_artifact
        
        except Exception as e : 
            raise MyException(e,sys) from e 
        

    def start_model_training(self,data_transformation_artfact :DataTransformationArtifact ) ->ModelTrainerArtifact:
        try : 
            logging.info("Entering into model training in pipeline")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artfact,model_trainer_config=self.model_trainer_config)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training done ")
            return model_trainer_artifact
        except Exception as e : 
            raise MyException(e,sys) from e 
        
    def start_model_evaluation(self,data_transformation_artifact : DataTransformationArtifact , model_trainer_artifact:ModelTrainerArtifact) ->ModelEvaluationArtifact: 
        try : 
            logging.info("Entering into model evaluation in pipeline")
            model_evaluation = ModelEvaluation(data_transformation_artifact=data_transformation_artifact , model_evaluation_config=self.model_evaluation_config, model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact 
        except Exception as e : 
            raise MyException(e,sys) from e 
        

    def start_model_pusher(self,model_evaluation_artifact : ModelEvaluationArtifact) : 
        try : 
            logging.info("Entering into model pusher in pipeline")
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,model_pusher_config=self.model_pusher_config) 
            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact 
        except Exception as e : 
            raise MyException(e,sys) from e


    def run_pipeline(self)->None : 
        try: 
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact,data_validation_artifact)
            model_trainer_artifact =self.start_model_training(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_transformation_artifact,model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted : 
                logging.info("Model is not accepted")
                return None 
            
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact)
        except Exception as e : 
            raise MyException(e,sys)

        
