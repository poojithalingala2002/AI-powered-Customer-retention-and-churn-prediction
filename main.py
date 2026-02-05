import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys   # <--- Make sure this line is here
import os
import seaborn as sns
import pickle
import warnings

from imbalance import balance_data

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from missing_values_techniques import random_sample_imputation_technique
from var import variable_transformation_outliers_minmax
from Feature_selection import  complete_feature_selection
from cat_to_num import cat
from imbalance import balance_data


from log_code import setup_logging
logger = setup_logging('main')


class CHURN_PREDICTION:
    def __init__(self, path):
        try:
            # Load dataset
            self.df = pd.read_csv(path)
            logger.info(f'Data loded successfully')
            logger.info(f'Total Rows in the data : {self.df.shape[0]}')
            logger.info(f'Total columns in the data : {self.df.shape[1]}')
            logger.info(f'Before : {self.df.isnull().sum()}')

            #self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
            self.df.drop(columns=['customerID'], inplace=True)

            # Split features and target
            self.X = self.df.iloc[:, :-2].join(self.df.iloc[:, -1])
            self.y = self.df.iloc[:, -2]

            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')

            logger.info(f'{self.y_train.sample(5)}')
            logger.info(f'{self.y_test.sample(5)}')

            logger.info(f'Training data size : {self.X_train.shape}')
            logger.info(f'Testing data size : {self.X_test.shape}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in init at line {error_line.tb_lineno}: {error_msg}")

    def missing_values_techniques(self):
        try:
            #logger.info(f'Total rows in training data : {self.X_train.shape}')
            logger.info(f"Before (Train):\n{self.X_train.isnull().sum().loc[lambda x: x > 0]}")
            logger.info(f"Before (Test):\n{self.X_test.isnull().sum().loc[lambda x: x > 0]}")
            self.X_train,self.X_test = random_sample_imputation_technique(self.X_train,self.X_test)
            logger.info(f'random sample technique applied')
            logger.info(f"After (Train):\n{self.X_train.isnull().sum().loc[lambda x: x > 0] if self.X_train.isnull().sum().any() else 0}")
            logger.info(f"After (Train):\n{self.X_test.isnull().sum().loc[lambda x: x > 0] if self.X_train.isnull().sum().any() else 0}")

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(f"Error in missing_values_techniques at line {error_line.tb_lineno}: {error_msg}")

    def out(self):
        try:
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')
            logger.info(f'----------------------------------')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f'{self.X_train_num.columns}')
            logger.info(f'{self.X_train_cat.columns}')
            logger.info(f'{self.X_test_num.columns}')
            logger.info(f'{self.X_test_cat.columns}')
            logger.info(f'{self.X_train_num.shape}')
            logger.info(f'{self.X_train_cat.shape}')
            logger.info(f'{self.X_test_num.shape}')
            logger.info(f'{self.X_test_cat.shape}')

            self.X_train_num, self.X_test_num = variable_transformation_outliers_minmax(self.X_train_num, self.X_test_num)

            logger.info(f"{self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f"{self.X_test_num.columns} -> {self.X_test_num.shape}")
        except Exception:
            error_type, error_msg, error_line = sys.exc_info()
            logger.error(
                f"Error in out at line {error_line.tb_lineno}: {error_msg}")

    def fs(self):
        try:
            logger.info(f'{self.X_train.columns}')
            logger.info(f'{self.X_test.columns}')
            logger.info(f'----------------------------------')
            self.X_train_num = self.X_train.select_dtypes(exclude='object')
            self.X_train_cat = self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')

            logger.info(f" Before: {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f" Before: {self.X_test_num.columns} -> {self.X_test_num.shape}")
            self.X_train_num, self.X_test_num = complete_feature_selection(self.X_train_num, self.X_test_num,self.y_train)
            logger.info(f" After: {self.X_train_num.columns} -> {self.X_train_num.shape}")
            logger.info(f" After: {self.X_test_num.columns} -> {self.X_test_num.shape}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no {error_line.tb_lineno}: due to {error_msg}')


    def cta_num(self):
        try:
            self.X_train_cat, self.X_test_cat = cat(self.X_train_cat, self.X_test_cat, self.X_train_num, self.X_test_num)

            self.training_data = pd.concat([self.X_train_num, self.X_train_cat], axis=1)
            self.testing_data = pd.concat([self.X_test_num, self.X_test_cat], axis=1)
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no {error_line.tb_lineno}: due to {error_msg}')

    def data(self):
        try:
            self.y_train = self.y_train.map({'Yes': 1, 'No': 0}).astype(int)
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)
            balance_data(self.training_data, self.y_train,self.testing_data, self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in line no {error_line.tb_lineno}: due to {error_msg}')


if __name__ == "__main__":
    try:
        data = 'Churn_Updated_set.csv'
        obj = CHURN_PREDICTION(data)
        obj.missing_values_techniques()
        obj.out()
        obj.fs()
        obj.cta_num()
        obj.data()


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error in main at line {error_line.tb_lineno}: {error_msg}")