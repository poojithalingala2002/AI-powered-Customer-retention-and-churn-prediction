import numpy as np
import pandas as pd
import sys
import warnings
import pickle

from log_code import setup_logging
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from all_models import common

warnings.filterwarnings('ignore')
logger = setup_logging('imbalance')


def balance_data(X_train, y_train, X_test, y_test):
    try:
        logger.info(f"Before Yes count : {sum(y_train == 1)}")
        logger.info(f"Before No count  : {sum(y_train == 0)}")

        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

        logger.info(f"After Yes count : {sum(y_train_bal == 1)}")
        logger.info(f"After No count  : {sum(y_train_bal == 0)}")

        # save feature columns
        with open("feature_columns.pkl", "wb") as f:
            pickle.dump(X_train_bal.columns.tolist(), f)

        # scale
        scaler = StandardScaler()
        scaler.fit(X_train_bal)

        X_train_scaled = scaler.transform(X_train_bal)
        X_test_scaled = scaler.transform(X_test)

        with open("scalar.pkl", "wb") as f:
            pickle.dump(scaler, f)

        common(X_train_scaled, y_train_bal, X_test_scaled, y_test)

    except Exception:
        error_type, error_msg, error_line = sys.exc_info()
        logger.error(f"Error at line {error_line.tb_lineno}: {error_msg}")
