import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model training started")
            logging.info("Splitting train and test data")
            x_train, y_train, x_test, y_test=(train_array[:,:-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1])
            
            models ={
                "Random_Forest":RandomForestRegressor(),
                "Decision_Tree":DecisionTreeRegressor(),
                "Gradient_Boosting":GradientBoostingRegressor(),
                "Linear_Regression":LinearRegression(),
                "K_Nearest_Neighbors":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBootRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
            }
            
            model_report:dict=evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            
            # get best score
            best_model_score=max(sorted(model_report.values()))
            
            # Get model name with best score
            best_model_name=list(model_report.keys())[
                                                      list(model_report.values()).index(best_model_score)
                                                      ]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("no best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            r2score=r2_score(y_test, predicted)
            
            return r2score
            
        except Exception as e:
            raise CustomException(e, sys)
        
