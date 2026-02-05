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
logger = setup_logging('var')
from scipy import stats

from sklearn.preprocessing import MinMaxScaler


def variable_transformation_outliers_minmax(X_train_num, X_test_num):
    try:
        logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
        logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

        scaler = MinMaxScaler()

        for i in X_train_num.columns:
            # Scale train
            X_train_scaled = scaler.fit_transform(X_train_num[[i]])
            X_train_num[i + "_minmax"] = X_train_scaled.flatten()
            X_train_num = X_train_num.drop([i], axis=1)

            # Calculate IQR limits on scaled train data
            q75 = X_train_num[i + "_minmax"].quantile(0.75)
            q25 = X_train_num[i + "_minmax"].quantile(0.25)
            iqr = q75 - q25
            upper_limit = q75 + 1.5 * iqr
            lower_limit = q25 - 1.5 * iqr

            # Trim outliers in train
            X_train_num[i + "_minmax_trim"] = np.where(
                X_train_num[i + "_minmax"] > upper_limit, upper_limit,
                np.where(X_train_num[i + "_minmax"] < lower_limit, lower_limit, X_train_num[i + "_minmax"])
            )
            X_train_num = X_train_num.drop([i + "_minmax"], axis=1)

            # Scale test using the same scaler
            X_test_scaled = scaler.transform(X_test_num[[i]])
            X_test_num[i + "_minmax_trim"] = np.where(
                X_test_scaled.flatten() > upper_limit, upper_limit,
                np.where(X_test_scaled.flatten() < lower_limit, lower_limit, X_test_scaled.flatten())
            )
            X_test_num = X_test_num.drop([i], axis=1)

        logger.info(f"{X_train_num.columns} -> {X_train_num.shape}")
        logger.info(f"{X_test_num.columns} -> {X_test_num.shape}")

        # **RETURN the transformed dataframes**
        return X_train_num, X_test_num

    except Exception as e:
        logger.error(f"Error in variable_transformation_outliers_minmax: {e}")
        # Return original data in case of error to avoid breaking pipeline
        return X_train_num, X_test_num
