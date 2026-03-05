import sys 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 

from src.constants import * 
from src.entity.config_entity import DataTransformationConfig 
from src.entity.artifacts_entity import DataIngestionArtifact , DataTransformationArtifact ,DataValidationArtifact
from src.exception import MyException 
from src.logging import logging 
from src.utils.main_utils import save_object , save_numpy_array_data 


class DataTransformation : 
    def __init__ (self ,data_ingestion_artifact : DataIngestionArtifact , data_validation_artifact : DataValidationArtifact ,  data_transformation_config : DataTransformationConfig) : 

        try : 
            self.data_ingestion_artifact = data_ingestion_artifact 
            self.data_validation_artifact = data_validation_artifact 
            self.data_transformation_config = data_transformation_config
        
        except Exception as e : 
            raise MyException(e,sys)
    
    @staticmethod 
    def read_data(file_path) -> pd.DataFrame : 
        try : 
            return pd.read_csv(file_path)
        except Exception as e : 
            raise MyException(e, sys)
        
    def extract_date_features(self,df:pd.DataFrame) ->pd.DataFrame : 
        df['day'] = df['date'].dt.day.astype('int8')
        df['month'] = df['date'].dt.month.astype('int8')
        df['year'] = df['date'].dt.year.astype('int16')
        df['day_of_week'] = df['date'].dt.dayofweek.astype('int8')
        df['quarter'] = df['date'].dt.quarter.astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_payday'] = df['date'].dt.is_month_end.astype('int8')
        df['time_index'] = (df['date'] - df['date'].min()).dt.days.astype('int32')
        return df
    
    def initiate_data_transformaion(self) -> DataTransformationArtifact : 
        try : 
            logging.info("Starting Data Trnsformation") 

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            oil_df = self.read_data(self.data_ingestion_artifact.oil_file_path)
            stores_df = self.read_data(self.data_ingestion_artifact.stores_file_path)
            holidays_df = self.read_data(self.data_ingestion_artifact.holiday_file_path)
            txn_df = self.read_data(self.data_ingestion_artifact.transactions_file_path)

            logging.info("Applying Base Merges ") 

            # A) Date formatting
            for df in [train_df,test_df,oil_df,stores_df,holidays_df,txn_df] : 
                if 'date' in df.columns : 
                    df['date']= pd.to_datetime(df['date'])

            # B) Clean Oil & Holidays

            oil_df['dcoilwtico'] = oil_df['dcoilwtico'].ffill().bfill()
            holidays_df = holidays_df[holidays_df['transferred'] == False].drop(columns=['transferred'])
            nat_holiday = holidays_df[holidays_df['locale'] == 'National'].rename(columns={'type': 'nat_holiday_type', 'description': 'nat_holiday_desc'})
            loc_reg_holidays = holidays_df[holidays_df['locale'].isin(['Regional', 'Local'])]

            # C) Combining Train & Test temporarily for uniform merging
            train_df['is_test'] = 0
            test_df['is_test'] = 1
            combined_df = pd.concat([train_df, test_df], ignore_index=True)

            stores_df = stores_df.rename(columns={'type': 'store_type'})

            # D) Merges

            combined_df = combined_df.merge(stores_df,on='store_nbr',how='left')
            combined_df = combined_df.merge(nat_holiday[['date', 'nat_holiday_type', 'nat_holiday_desc']], on='date', how='left')
            combined_df = combined_df.merge(
                loc_reg_holidays[['date', 'locale_name', 'type', 'description']],
                left_on=['date', 'city'], right_on=['date', 'locale_name'], how='left'
            )
            combined_df = combined_df.merge(txn_df, on=['date', 'store_nbr'], how='left')
            combined_df = combined_df.merge(oil_df, on='date', how='left')

            # E) Clean Merged Columns

            combined_df['transactions'] = combined_df['transactions'].fillna(0)
            cols_to_fill = ['nat_holiday_type', 'locale_name', 'type', 'description', 'nat_holiday_desc']
            combined_df[cols_to_fill] = combined_df[cols_to_fill].fillna('None')
            
            combined_df['is_holiday'] = ((combined_df['nat_holiday_type'] != 'None') | (combined_df['type'] != 'None')).astype(int)
            combined_df['desc'] = np.where(combined_df['nat_holiday_desc'] != 'None', combined_df['nat_holiday_desc'], combined_df['description'])
            
            combined_df = combined_df.drop(columns=['nat_holiday_desc', 'description', 'locale_name'])
            combined_df = self.extract_date_features(combined_df)

            # F) Log transform target variable
            combined_df['sales'] = np.log1p(combined_df['sales'])

            # Split back 
            train_df = combined_df[combined_df['is_test'] == 0].copy().drop(columns=['is_test'])
            test_df = combined_df[combined_df['is_test'] == 1].copy().drop(columns=['is_test'])

            logging.info("Calculating Target-Based Features (Strictly on Train to avoid Leakage)")

            #### TARGET BASED FEATURES ####

            # A) Holiday Lift Analysis

            baseline_sales = train_df[train_df['desc'] == 'None']['sales'].mean()
            holiday_stats = train_df.groupby('desc')['sales'].mean().reset_index()
            holiday_stats['lift'] = holiday_stats['sales'] / baseline_sales
            big_events = holiday_stats[holiday_stats['lift'] >= 1.30]['desc'].unique().tolist()
            big_events = [e for e in big_events if e != 'None']

            train_df['is_big_event'] = train_df['desc'].isin(big_events).astype(int)
            test_df['is_big_event'] = test_df['desc'].isin(big_events).astype(int)

            # B) Family Avg Sales

            family_mapping = train_df.groupby('family')['sales'].mean().to_dict()
            global_mean = train_df['sales'].mean()
            train_df['family_avg_sales'] = train_df['family'].map(family_mapping)
            test_df['family_avg_sales'] = test_df['family'].map(family_mapping).fillna(global_mean)

            # C) Transaction Backup
            txn_mapping = train_df.groupby(['store_nbr', 'day_of_week'])['transactions'].mean().reset_index()
            txn_mapping.rename(columns={'transactions': 'txn_backup'}, inplace=True)

            # D) Label Encoding
            cat_features = ['city', 'state', 'store_type', 'cluster', 'type', 'nat_holiday_type', 'family', 'desc']
            preprocessing_obj = {'family_mapping': family_mapping, 'big_events': big_events, 'label_encoders': {}}


            for col in cat_features:
                if col in train_df.columns:
                    le = LabelEncoder()
                    train_df[col] = le.fit_transform(train_df[col].astype(str))
                    
                    # Handle unseen test labels
                    test_df[col] = test_df[col].astype(str).map(lambda s: s if s in le.classes_ else '<unknown>')
                    le.classes_ = np.append(le.classes_, '<unknown>')
                    test_df[col] = le.transform(test_df[col])
                    
                    preprocessing_obj['label_encoders'][col] = le

            logging.info("Calculating Lag Features...")


            ### AGAIN TEMP COMBINIG FOR LAGS 

            combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

            # Standard Sales Lags
            for i in [1, 7, 14, 30]:
                combined_df[f'sales_lag_{i}'] = combined_df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(i))

            # Rolling Mean
            combined_df['rolling_mean_7'] = combined_df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(1).rolling(window=7).mean())


            # Oil & Txn Lags
            temp_oil = combined_df[['date', 'dcoilwtico']].drop_duplicates().sort_values('date')
            temp_oil['oil_price_lag_1'] = temp_oil['dcoilwtico'].shift(1)
            temp_oil['oil_price_diff'] = temp_oil['dcoilwtico'] - temp_oil['oil_price_lag_1']
            combined_df = combined_df.merge(temp_oil[['date', 'oil_price_lag_1', 'oil_price_diff']], on='date', how='left')
            
            combined_df['transactions_lag_1'] = combined_df.groupby(['store_nbr', 'family'])['transactions'].transform(lambda x: x.shift(1))

            # Split back final time based on cutoff date
            cutoff_date = train_df['date'].max()
            train_df = combined_df[combined_df['date'] <= cutoff_date].copy()
            test_df = combined_df[combined_df['date'] > cutoff_date].copy()

            # Fill NA for missing Test transactions using backup
            test_df = test_df.merge(txn_mapping, on=['store_nbr', 'day_of_week'], how='left')
            test_df['transactions_lag_1'] = test_df['transactions_lag_1'].fillna(test_df['txn_backup'])
            test_df.drop(columns=['txn_backup'], inplace=True)

            train_df = train_df.fillna(0)
            test_df = test_df.fillna(0)


            drop_cols = ['date', 'transactions', 'dcoilwtico', 'desc']
            train_df = train_df.drop(columns=drop_cols, errors='ignore')
            test_df = test_df.drop(columns=drop_cols, errors='ignore')

            target_col = 'sales'
            train_cols = [c for c in train_df.columns if c != target_col] + [target_col]
            test_cols = [c for c in test_df.columns if c != target_col] + [target_col]

            target_col = 'sales'
            train_cols = [c for c in train_df.columns if c != target_col] + [target_col]
            test_cols = [c for c in test_df.columns if c != target_col] + [target_col]

        
            print("\n" + "="*50)
            print("🚀 EXACT COLUMN ORDER FOR PREDICTION:")
            print(train_cols)
            print("="*50 + "\n")
         


            train_arr = train_df[train_cols].values
            test_arr = test_df[test_cols].values

            logging.info("Saving numpy arrays and preprocessing object...")
            
            save_numpy_array_data(self.data_transformation_config.transformaion_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformaion_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )


            logging.info(f"Data transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)

