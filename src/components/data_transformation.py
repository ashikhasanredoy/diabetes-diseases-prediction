import sys
import os
from dataclasses import dataclasses
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclasses
class DataTransformationConfig:
    preprocessor_obj_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def data_transformation_obj(self):
        try:
            numeric_columns=['Pregnancies',
                         'Glucose',
                         'BloodPressure',
                         'SkinThickness',
                         'Insulin',
                         'BMI',
                         'DiabetesPedigreeFunction',
                         'Age',
                         'Outcome']
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    
                ]
            )
            
            logging.info(f"all numeric columns{numeric_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ('num pipeline',num_pipeline,numeric_columns)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException (e,sys)  
        
        
    def data_tranfromation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("reading the train and test data")
            preprocessor_obj=self.data_transformation_obj()
            target_colums_name='Outcome'
            
            input_feature_train_df=train_df.drop(columns=[target_colums_name])
            traget_feature_train_df=train_df[target_colums_name]
            
            input_feature_test_df=test_df.drop(columns=[target_colums_name])
            target_feature_test_df=test_df[target_colums_name]
            
            logging.info("Applying preproceesor on the data set")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr=np.c_(input_feature_train_arr,np.array(traget_feature_train_df))
            test_arr=np.c_(input_feature_test_arr,np.array(target_feature_test_df))
            
            logging.info("Saving all array")
            
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                onj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
            
        except Exception as e:
            raise CustomException (e,sys)       