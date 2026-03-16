import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,feature):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(feature)
            pred=model.predict(data_scaled)
            
            return(pred)
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 Pregnancies:int,
                 Glucose:int,
                 BloodPressure:int,
                 SkinThickness:int,
                 Insulin:int,
                 BMI:float,
                 DiabetesPedigreeFunction:float,
                 Age:int)  :
        self.Pregnancies=Pregnancies
        self.Glucose=Glucose
        self.BloodPressure=BloodPressure
        self.SkinThickness=SkinThickness
        self.Insulin=Insulin
        self.BMI=BMI
        self.DiabetesPedigreeFunction=DiabetesPedigreeFunction
        self.Age=Age
        
        
    def data_frame(self):
        try:
            input_data_dict={
                "Pregnancies":[self.Pregnancies],
                "Glucose":[self.Glucose],
                "BloodPressure":[self.BloodPressure],
                "SkinThickness":[self.SkinThickness],
                "Insulin":[self.Insulin],
                "BMI":[self.BMI],
                "DiabetesPedigreeFunction":[self.DiabetesPedigreeFunction],
                "Age":[self.Age]
            }
            
            return pd.DataFrame(input_data_dict)
        except Exception as e:
            raise  CustomException(e,sys)