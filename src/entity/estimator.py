import numpy as np
import pandas as pd

class MyModel : 
    
    def __init__(self,preprocessing_object, trained_model_object) : 
        self.preprocessing_object = preprocessing_object 
        self.trained_model_object = trained_model_object 
    
    def predict(self,raw_dataframe) : 
        df = raw_dataframe.copy()

        family_map = self.preprocessing_object['family_mapping']
        df['family_avg_sales'] = df['family'].map(family_map)

        big_events_list = self.preprocessing_object['big_events']
        df['is_big_event'] = df['desc'].isin(big_events_list).astype(int)

        label_encoders_dict = self.preprocessing_object['label_encoders']


        

        for col , le in label_encoders_dict.items() : 
            if col in df.columns : 
                df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else '<unknown>')
                df[col] = le.transform(df[col])

        cols_to_drop = ['date', 'desc', 'transactions', 'dcoilwtico','sales']
        df = df.drop(columns=cols_to_drop, errors='ignore')


        expected_cols = [
            'store_nbr', 'family', 'onpromotion', 'city', 'state', 'store_type', 
            'cluster', 'nat_holiday_type', 'type', 'is_holiday', 'day', 'month', 
            'year', 'day_of_week', 'quarter', 'is_weekend', 'is_payday', 
            'time_index', 'is_big_event', 'family_avg_sales', 'sales_lag_1', 
            'sales_lag_7', 'sales_lag_14', 'sales_lag_30', 'rolling_mean_7', 
            'oil_price_lag_1', 'oil_price_diff', 'transactions_lag_1'
        ]
        df = df[expected_cols]
        
   
        input_array = df.values
        predictions_log = self.trained_model_object.predict(input_array)
        
      
        predictions_actual = np.expm1(predictions_log)
        
        return predictions_actual


     