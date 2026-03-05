import sys 
import pandas as pd 
import numpy as np
import os
import pymongo

from src.entity.config_entity import PredictionPipelineConfig
from src.entity.s3_estimator import ProjEstimator 
from src.exception import MyException
from src.logging import logging
from src.utils.main_utils import load_object
import dotenv
from dotenv import load_dotenv

load_dotenv()



class SalesData : 
    def __init__(self ,date: str,
                 store_nbr: int,
                 family: str,
                 onpromotion: int,
                 dcoilwtico: float,
                 city: str,
                 state: str,
                 store_type: str,
                 cluster: int,
                 transactions: int = None, 
                 holiday_type: str = 'None',
                 holiday_description: str = 'None') :
        try : 
            self.date = date
            self.store_nbr = store_nbr
            self.family = family
            self.onpromotion = onpromotion
            self.dcoilwtico = dcoilwtico
            self.city = city
            self.state = state
            self.store_type = store_type
            self.cluster = cluster
            self.transactions = transactions
            self.holiday_type = holiday_type
            self.holiday_description = holiday_description
        except Exception as e : 
            raise MyException(e,sys) from e
        
    
    def get_sales_data_as_dict(self) : 
        logging.info("formatting raw data input into dict")

        try : 
            input_data = {
                "date": [self.date],
                "store_nbr": [self.store_nbr],
                "family": [self.family],
                "onpromotion": [self.onpromotion],
                "dcoilwtico": [self.dcoilwtico],
                "city": [self.city],
                "state": [self.state],
                "store_type": [self.store_type], 
                "cluster": [self.cluster],
                "transactions": [self.transactions],
                "nat_holiday_type": [self.holiday_type],
                "type": ['None'], 
                "desc": [self.holiday_description]
            }
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e
        
    
    def get_sales_input_dataframe(self) -> pd.DataFrame : 
        try : 
            input_dict = self.get_sales_data_as_dict() 
            df = pd.DataFrame(input_dict)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception as e : 
            raise MyException(e,sys) from e 
        
    
class SalesPredictor : 
    def __init__(self,prediction_config : PredictionPipelineConfig = PredictionPipelineConfig()) : 
        try:
            self.config = prediction_config
        except Exception as e:
            raise MyException(e, sys)
        
    def fetch_historical_context(self, target_date: pd.Timestamp, store_nbr: int, family: str) -> pd.DataFrame: 
        """
        Production Step: Fetch exactly past 30 days of real data from MongoDB for Lags.
        """
        try:
            logging.info(f"Fetching real historical context for Store: {store_nbr}, Family: {family} prior to {target_date}")
            
            start_date = target_date - pd.Timedelta(days=30)
            end_date = target_date - pd.Timedelta(days=1)
            
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            mongo_db_url = os.getenv("MONGO_URL") 
            if mongo_db_url is None:
                raise Exception("Environment variable 'MONGO_URL' is not set.")
                
            client = pymongo.MongoClient(mongo_db_url)
            
            database = client['Proj1'] 
            sales_collection = database['Train_Data'] 
            txn_collection = database['Transactions']

  
            clean_store_nbr = int(store_nbr)
            query = {
                "store_nbr": clean_store_nbr,
                "family": family,
                "date": {
                    "$gte": start_date_str,
                    "$lte": end_date_str
                }
            }

     
            sales_cursor = sales_collection.find(
                query, 
                {"_id": 0, "date": 1, "store_nbr": 1, "family": 1, "sales": 1}
            ).sort("date", 1)

            txn_query = {
                "store_nbr": clean_store_nbr,
                "date": {"$gte": start_date_str, "$lte": end_date_str}
            }

            txn_cursor = txn_collection.find(txn_query, {"_id": 0, "date": 1, "store_nbr": 1, "transactions": 1})
            txn_df = pd.DataFrame(list(txn_cursor))
            
       
            history_df = pd.DataFrame(list(sales_cursor))
            
            # Backup Plan (Agar kisi naye store ka data na ho DB mein)
            if history_df.empty:
                logging.warning(f"No history found in DB for Store {store_nbr}, Family {family}. Padding with zeros to avoid pipeline crash.")
               
                dates = pd.date_range(start=start_date, end=end_date)
                history_df = pd.DataFrame({
                    'date': dates,
                    'store_nbr': clean_store_nbr,
                    'family': family,
                    'sales': 0.0, 
                    'transactions': 0
                })
            else:
                logging.info("History found in DB for Store {store_nbr}, Family {family}")
                
                history_df['date'] = pd.to_datetime(history_df['date'])
                if not txn_df.empty:
                    txn_df['date'] = pd.to_datetime(txn_df['date'])
                    
                    # Merging transactions based on Date and Store
                    history_df = history_df.merge(txn_df, on=['date', 'store_nbr'], how='left')
                    
                  
                    history_df['transactions'] = history_df['transactions'].fillna(0)
                else:
                    logging.warning(f"No transactions found for Store {store_nbr}. Defaulting to 0.")
                    history_df['transactions'] = 0


            logging.info(f"Successfully fetched {len(history_df)} days of historical data from DB.")
            return history_df

        except Exception as e:
            raise MyException(e, sys) from e 

    def calculate_dynamic_features(self, current_input_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates lags based on the historical context.
        """
        logging.info("Calculating dynamic lag features for prediction.")
        
        
        current_input_df['sales'] = np.nan 

        history_df['sales_log'] = np.log1p(history_df['sales'])
        
        # Create a unified dataframe for calculating rolling/shift metrics
        # only need 'date', 'sales', and 'transactions' from history for lags
        history_subset = history_df[['date', 'sales', 'transactions']].copy()
        
    
        # --- Lag Calculation (Using the LOGGED sales now) ---
        current_input_df['sales_lag_1'] = history_df['sales_log'].iloc[-1]
        current_input_df['sales_lag_7'] = history_df['sales_log'].iloc[-7] if len(history_df) >= 7 else history_df['sales_log'].mean()
        current_input_df['sales_lag_14'] = history_df['sales_log'].iloc[-14] if len(history_df) >= 14 else history_df['sales_log'].mean()
        current_input_df['sales_lag_30'] = history_df['sales_log'].iloc[-30] if len(history_df) >= 30 else history_df['sales_log'].mean()
        
        current_input_df['rolling_mean_7'] = history_df['sales_log'].tail(7).mean()

        current_input_df['transactions_lag_1'] = history_df['transactions'].iloc[-1]
        
        # Added basic date features needed by MyModel
        current_input_df['day'] = current_input_df['date'].dt.day.astype('int8')
        current_input_df['month'] = current_input_df['date'].dt.month.astype('int8')
        current_input_df['year'] = current_input_df['date'].dt.year.astype('int16')
        current_input_df['day_of_week'] = current_input_df['date'].dt.dayofweek.astype('int8')
        current_input_df['quarter'] = current_input_df['date'].dt.quarter.astype('int8')
        current_input_df['is_weekend'] = (current_input_df['day_of_week'] >= 5).astype('int8')
        current_input_df['is_payday'] = current_input_df['date'].dt.is_month_end.astype('int8')

        current_input_df['is_holiday'] = ((current_input_df['nat_holiday_type'] != 'None') | (current_input_df['type'] != 'None')).astype(int)
        
       
        current_input_df['time_index'] = (current_input_df['date'] - pd.to_datetime('2013-01-01')).dt.days.astype('int32')
        
        
        current_input_df['oil_price_lag_1'] = current_input_df['dcoilwtico']
        current_input_df['oil_price_diff'] = 0.0
        
        return current_input_df

    def predict(self, raw_dataframe: pd.DataFrame) -> float:
        """
        The main prediction orchestrator.
        """
        try:
            logging.info("Starting prediction orchestration.")
            
            target_date = raw_dataframe['date'].iloc[0]
            store = raw_dataframe['store_nbr'].iloc[0]
            family = raw_dataframe['family'].iloc[0]

            #  Fetch History
            history_df = self.fetch_historical_context(target_date, store, family)
            
            #  Generate Lags
            enriched_df = self.calculate_dynamic_features(raw_dataframe.copy(), history_df)

          
            logging.info("Loading production model via S3 Estimator.")
            s3_estimator = ProjEstimator(
                bucket_name=self.config.bucket_name,
                model_path=self.config.model_file_path
            )
            
       
            predicted_sales = s3_estimator.predict(enriched_df)
            
            logging.info(f"Prediction complete. Forecasted sales: {predicted_sales[0]}")
            return predicted_sales[0]

        except Exception as e:
            raise MyException(e, sys)

