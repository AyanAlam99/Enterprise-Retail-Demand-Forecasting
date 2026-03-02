import sys 
from typing import Tuple 
import numpy as np 
import xgboost as xgb 
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

from src.utils.main_utils import load_numpy_array_data , load_object,save_object , read_yaml
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifacts_entity import DataTransformationArtifact, RegressionMetricArtifact , ModelTrainerArtifact
from src.exception import MyException 
from src.logging import logging
from src.entity.estimator import MyModel


class ModelTrainer : 
    def __init__(self,data_transformation_artifact : DataTransformationArtifact , model_trainer_config : ModelTrainerConfig) :
        self.data_transformation_artifact = data_transformation_artifact 
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train:np.array, test: np.array) -> Tuple[object,object,object] : 

        try : 
            logging.info("Splitting the train and test data into features and target variable")
            x_train  , y_train = train[:,:-1] , train[:,-1]
            x_test, y_test = test[:, :-1], test[:, -1]
            logging.info("train-test split done.")

            logging.info("reading model hyperparams from model yaml file ")
            model_config = read_yaml(self.model_trainer_config.model_config_file_path)
            xgb_params = model_config['model_selection']['model']['XGBRegressor']

            logging.info("Initializing XGBRegressor with specified parameters")
            model = xgb.XGBRegressor(**xgb_params)

            logging.info("Model training starts")
            model.fit(
                x_train,y_train,
                eval_set = [(x_train,y_train),(x_test,y_test)],
                verbose = 100
            )
            logging.info("Model training done.")

            y_train_pred_log = model.predict(x_train)
            y_test_pred_log = model.predict(x_test)

            y_tain_pre_clipped = np.clip(y_train_pred_log,a_min= 0 , a_max = 20)
            y_test_pre_clipped = np.clip(y_test_pred_log,a_min= 0 , a_max = 20)

            y_train_actual = np.expm1(y_train)
            y_test_actual = np.expm1(y_test)
            y_train_pred_actual = np.expm1(y_tain_pre_clipped)
            y_test_pred_actual = np.expm1(y_test_pre_clipped)


            train_r2 = r2_score(y_train_actual,y_train_pred_actual)
            train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
            train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)

            test_r2 = r2_score(y_test_actual, y_test_pred_actual)
            test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
            test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)

            train_metric_artifact = RegressionMetricArtifact(r2_score=train_r2, rmse=train_rmse, mae=train_mae)
            test_metric_artifact = RegressionMetricArtifact(r2_score=test_r2, rmse=test_rmse, mae=test_mae)

            return model , train_metric_artifact , test_metric_artifact 
        
        except Exception as e : 
            raise MyException(e,sys) from e 
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact : 

        try : 
            print("Starting Model Trainer Component")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")

            trained_model , train_metric , test_metric = self.get_model_object_and_report(train=train_arr , test= test_arr)
            logging.info(f"Model trained. Train R2: {train_metric.r2_score:.4f}, Test R2: {test_metric.r2_score:.4f}")

            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            if test_metric.r2_score < self.model_trainer_config.expected_score : 
                logging.info(f"No model found with score above the base score: {self.model_trainer_config.expected_accuracy}")
                raise Exception(f"Model Failed. Test R2 {test_metric.r2_score} is below expected accuracy {self.model_trainer_config.expected_accuracy}")
            
            diff = abs(train_metric.r2_score - test_metric.r2_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.info(f"Model Overfitting Check Failed. Difference: {diff}")
                raise Exception(f"Model Overfitting! Difference {diff} is greater than threshold {self.model_trainer_config.overfitting_underfitting_threshold}")


            logging.info("Model passed quality gates. Saving new model.")
            my_model = MyModel(preprocessing_object=preprocessing_obj,trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path ,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

