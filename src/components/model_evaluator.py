import sys , os 
import numpy as np 
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import r2_score

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifacts_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from src.entity.s3_estimator import Proj1Estimator  
from src.exception import MyException
from src.logging import logging
from src.utils.main_utils import load_numpy_array_data

@dataclass 
class EvaluateModelResponse : 
    trained_model_r2_score : float 
    best_model_r2_score : float 
    is_model_accepted : bool 
    difference : float 


class ModelEvaluation : 
    def __init__(self,data_transformation_artifact : DataTransformationArtifact , model_evaluation_config:ModelEvaluationConfig,model_trainer_artifact: ModelTrainerArtifact) : 

        try : 
            self.model_evaluation_config = model_evaluation_config 
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e : 
            raise MyException(e,sys) from e 
    
    def get_best_model(self)->Optional[Proj1Estimator] : 
        try : 
            bucket_name = self.model_evaluation_config.bucket_name
            model_path = self.model_evaluation_config.s3_model_key_path
            s3_estimator = Proj1Estimator(bucket_name,model_path)
            if s3_estimator.is_model_present(model_path=model_path):
                return s3_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)
        
    def evaluate_model(self) ->EvaluateModelResponse : 
        try : 
            logging.info("Loading transformed test array for evaluation...")
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            x_test,y_test = test_arr[:,:-1] , test_arr[:,-1]
            y_test_actual = np.exmp1(y_test)

            trained_model_r2_score = self.model_trainer_artifact.test_metric_artifact.r2_score
            logging.info(f"R2 Score for Newly Trained Model: {trained_model_r2_score}")

            best_model_r2_score = None
            logging.info("Fetching production model from S3 (if available)...")
            best_model = self.get_best_model()

            if best_model is not  None : 
                logging.info("Production model found. Computing R2 Score for production model...")
                y_hat_best_model_log = best_model.predict(x_test)
                y_hat_best_model_clipped = np.clip(y_hat_best_model_log, a_min=0, a_max=20)
                y_hat_best_model_actual = np.expm1(y_hat_best_model_clipped)

                best_model_r2_score = r2_score(y_test_actual , y_hat_best_model_actual)

                logging.info(f"R2 Score - Production Model: {best_model_r2_score} | Newly Trained Model: {trained_model_r2_score}")
            else: 
                logging.info("No production model found in S3. The newly trained model is the first of its kind.")
            
            tmp_best_model_score = 0 if best_model_r2_score is None else best_model_r2_score

            result = EvaluateModelResponse(
                trained_model_r2_score,
                best_model_r2_score,
                is_model_accepted =trained_model_r2_score > tmp_best_model_score,
                difference=trained_model_r2_score - tmp_best_model_score
            )
            logging.info(f"Model Evaluation Result: {result}")
            return result
        except Exception as e:
            raise MyException(e, sys)
        
    
    def initiate_model_evaluation (self) -> ModelEvaluationArtifact : 
        try : 
            logging.info("Starting Model Evaluation Component")
            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                changed_score = evaluate_model_response.difference ,
                s3_model_path = self.model_evaluation_config.s3_model_key_path,
                trained_model_path = self.model_trainer_artifact.trained_model_file_path
            )
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e