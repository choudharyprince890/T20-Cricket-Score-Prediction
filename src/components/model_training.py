import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from src.utils import save_object
from src.utils import evaluate_models



@dataclass
class ModelTraningConfig:
    trained_model_file_path = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_trainer_config = ModelTraningConfig()

    def initatied_model_traning(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
            }
            
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            ## To get best model value from models dictionary
            best_model_score = max(sorted(model_report.values()))

            ## To get best model key from models dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            # setting the threashold for best model
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

            
        except Exception as e:
            raise CustomException(e,sys)