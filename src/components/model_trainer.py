import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from src.exception import CustomException
from src.utils import evaluate_model,save_obj
from src.logger import logging
from dataclasses import dataclass


@dataclass 
class ModelTrainingConfig:
    trained_model_path_config=os.path.join('artifacts','model.pkl')
   
class ModelTraining:
    def __init__(self):
        self.model_train_config=ModelTrainingConfig()   
        
    def initiate_model_trained(self,train_arr,test_arr):
        
        try:
            logging.info("Spliting the train and test input")
            
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models={
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'RandomForestClassifier':RandomForestClassifier(),
                'AdaBoostClassifier':AdaBoostClassifier(),
                'GradientBoostingClassifier':GradientBoostingClassifier(),
                'SVC':SVC(),
                'LogisticRegression':LogisticRegression(),
                'KNeighborsClassifier':KNeighborsClassifier()
            }
            params={
                'DecisionTreeClassifier':{
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'RandomForestClassifier':{
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                },
                'AdaBoostClassifier':{
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
                    'algorithm': ['SAMME']
                },
                'GradientBoostingClassifier':{
                    'loss': ['log_loss', 'exponential'],
                    'learning_rate': [0.001, 0.01, 0.05, 0.1],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'max_depth': [3, 5, 10],
                    'max_features': ['sqrt', 'log2', None]
                },
                'SVC':{
                    'C': [0.01, 0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto'],
                    'probability': [True]
                },
                "LogisticRegression": {
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'solver': ['saga'],
                    'max_iter': [100, 200, 500]
                },

                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'metric': ['minkowski', 'euclidean', 'manhattan']
                }
            }
            model_report:dict=evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score < 0.7:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name} with accuracy score: {best_model_score}")
            
            save_obj(
                file_path=self.model_train_config.trained_model_path_config,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            score=accuracy_score(y_test,predicted)
            return score
        except Exception as e:
            raise CustomException(e,sys)     