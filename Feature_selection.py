import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('Feature_selection')
from scipy import stats
from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold
reg_con = VarianceThreshold(threshold=0.0)
reg_quasi = VarianceThreshold(threshold=0.1)

def complete_feature_selection(X_train_num, X_test_num, y_train):
    try:
        logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
        logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

        #constant technique
        reg_con.fit(X_train_num)
        logger.info(f'columns we need to remove from constant technique: {X_test_num.columns[~reg_con.get_support()]}')
        good_data = reg_con.transform(X_train_num)
        good_data_1 = reg_con.transform(X_test_num)
        X_train_num_fs = pd.DataFrame(data=good_data, columns=X_train_num.columns[reg_con.get_support()])
        X_test_num_fs = pd.DataFrame(data=good_data_1, columns=X_test_num.columns[reg_con.get_support()])

        # quaasi technique
        reg_quasi.fit(X_train_num_fs)
        logger.info(f'columns we need to remove from constant technique: {X_test_num_fs.columns[~reg_quasi.get_support()]}')
        good_data_2 = reg_quasi.transform(X_train_num_fs)
        good_data_3 = reg_quasi.transform(X_test_num_fs)
        X_train_num_fs_1 = pd.DataFrame(data=good_data_2, columns=X_train_num_fs.columns[reg_quasi.get_support()])
        X_test_num_fs_2 = pd.DataFrame(data=good_data_3, columns=X_test_num_fs.columns[reg_quasi.get_support()])

        #hypothesis testing
        logger.info(f"{X_train_num_fs_1.columns} -> {X_train_num_fs_1.shape}")
        logger.info(f"{X_test_num_fs_2.columns} -> {X_test_num_fs_2.shape}")
        logger.info(f'{y_train.unique()}')
        y_train = y_train.map({'Yes': 1, 'No': 0}).astype(int)
        logger.info(f'{y_train.unique()}')

        values = []
        plt.figure(figsize=(5, 3))
        for i in X_train_num_fs_1.columns:
            values.append(pearsonr(X_train_num_fs_1[i], y_train))
        values = np.array(values)
        p_values = pd.Series(values[:, 1], index=X_train_num_fs_1.columns)
        p_values.sort_values(ascending=False, inplace=True)
        logger.info(f'{p_values}')
        X_train_num_fs_1 = X_train_num_fs_1.drop(['SeniorCitizen'], axis=1)
        X_test_num_fs_2 = X_test_num_fs_2.drop(['SeniorCitizen'], axis=1)
        logger.info(f"{X_train_num_fs_1.columns} -> {X_train_num_fs_1.shape}")
        logger.info(f"{X_test_num_fs_2.columns} -> {X_test_num_fs_2.shape}")

        return X_train_num_fs_1, X_test_num_fs_2

    except Exception as e:
        error_type,error_msg,error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')