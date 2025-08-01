import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer # column transformer is used for pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact', "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTranformationConfig()
        
    def get_transformer_object(self):
        '''
        This function is responsible for getting data preprocesser
        '''
        try:
            numerical_columns=["writing_score", "reading_score"]
            categorical_columns= ["gender",
                                  "race_ethnicity",
                                  "parental_level_of_education", 
                                  "lunch",
                                  "test_preparation_course"]
            
            #PIpeline
            num_pipeline=Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), 
                       ("scaler", StandardScaler(with_mean=False))
                       ]
            )
            logging.info("Numerical columns encoding defined")
            
            cat_pipeline=Pipeline(
                steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                       ("one_hot_encoder", OneHotEncoder()),
                       ("scaler", StandardScaler(with_mean=False))
                       ]
            )
            logging.info("Categorical columns encoding defined")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            logging.info("preprocessor defined")
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        
        '''
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Train/Test data read completed")
            logging.info("Getting preprocessor object")
            
            preprocessor_obj=self.get_transformer_object()
            target_column="math_score"
            numerical_cols=["writing_score", "reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df=test_df[target_column]
            
            logging.info("Applying preprocessor on training and test dataset")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr,  np.array(target_feature_test_df)
            ]
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)