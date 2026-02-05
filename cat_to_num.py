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
logger = setup_logging('cat_to_num')
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def cat(X_train_cat, X_test_cat, X_train_num, X_test_num):
    try:
        logger.info(f'{X_train_cat.columns}')
        logger.info(f'{X_test_cat.columns}')

        for i in X_train_cat.columns:
            logger.info(f"{i} -> : {X_train_cat[i].unique()}")

        logger.info(f"Before Converting : {X_train_cat.columns}")
        logger.info(f"Before Converting : {X_test_cat.columns}")

        # ✅ FIX: handle missing SIM_Provider (Flask input case)
        if 'SIM_Provider' not in X_train_cat.columns:
            X_train_cat['SIM_Provider'] = 'Unknown'
        if 'SIM_Provider' not in X_test_cat.columns:
            X_test_cat['SIM_Provider'] = 'Unknown'

        one_hot = OneHotEncoder(drop='first', handle_unknown='ignore')

        one_hot.fit(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                 'PaperlessBilling', 'PaymentMethod', 'SIM_Provider']])

        result = one_hot.transform(X_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                                'TechSupport', 'StreamingTV', 'StreamingMovies',
                                                'PaperlessBilling', 'PaymentMethod', 'SIM_Provider']]).toarray()

        f = pd.DataFrame(data=result, columns=one_hot.get_feature_names_out())
        X_train_cat.reset_index(drop=True, inplace=True)
        f.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, f], axis=1)

        X_train_cat = X_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                        'TechSupport', 'StreamingTV', 'StreamingMovies',
                                        'PaperlessBilling', 'PaymentMethod', 'SIM_Provider'], axis=1)

        result1 = one_hot.transform(X_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                                'TechSupport', 'StreamingTV', 'StreamingMovies',
                                                'PaperlessBilling', 'PaymentMethod', 'SIM_Provider']]).toarray()

        f1 = pd.DataFrame(data=result1, columns=one_hot.get_feature_names_out())
        X_test_cat.reset_index(drop=True, inplace=True)
        f1.reset_index(drop=True, inplace=True)
        X_test_cat = pd.concat([X_test_cat, f1], axis=1)

        X_test_cat = X_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                      'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                      'TechSupport', 'StreamingTV', 'StreamingMovies',
                                      'PaperlessBilling', 'PaymentMethod', 'SIM_Provider'], axis=1)

        # 🔧 Ordinal Encoder (already correct)
        ord_end = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ord_end.fit(X_train_cat[['Contract']])

        result2 = ord_end.transform(X_train_cat[['Contract']])
        t = pd.DataFrame(data=result2, columns=ord_end.get_feature_names_out() + '_res')
        X_train_cat.reset_index(drop=True, inplace=True)
        t.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, t], axis=1)
        X_train_cat = X_train_cat.drop(['Contract'], axis=1)

        result3 = ord_end.transform(X_test_cat[['Contract']])
        t1 = pd.DataFrame(data=result3, columns=ord_end.get_feature_names_out() + '_res')
        X_test_cat.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        X_test_cat = pd.concat([X_test_cat, t1], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)

        X_train_num.reset_index(drop=True, inplace=True)
        X_test_num.reset_index(drop=True, inplace=True)

        training_data = pd.concat([X_train_num, X_train_cat], axis=1)
        testing_data = pd.concat([X_test_num, X_test_cat], axis=1)

        logger.info(f"After Converting : {X_train_cat.columns}")
        logger.info(f"After Converting : {X_test_cat.columns}")

        logger.info(f'{training_data.shape}')
        logger.info(f'{testing_data.shape}')

        logger.info(f'{training_data.isnull().sum()}')
        logger.info(f'{testing_data.isnull().sum()}')

        logger.info(f"=======================================================")

        for i in X_train_cat.columns:
            logger.info(f"{i} -> : {X_train_cat[i].unique()}")

        return training_data, testing_data

    except Exception:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'error in line no:{error_line.tb_lineno}:due to {error_msg}')
        raise
