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
logger = setup_logging('all_models')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
import xgboost as xgb

import pickle

def knn(X_train,y_train,X_test,y_test):
    try:
      global knn_reg
      knn_reg = KNeighborsClassifier(n_neighbors=5)
      knn_reg.fit(X_train,y_train)
      logger.info(f'KNN Test Accuracy : {accuracy_score(y_test,knn_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def nb(X_train,y_train,X_test,y_test):
    try:
      global naive_reg
      naive_reg = GaussianNB()
      naive_reg.fit(X_train,y_train)
      logger.info(f'Naive Bayes Test Accuracy : {accuracy_score(y_test,naive_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def lr(X_train,y_train,X_test,y_test):
    try:
      global lr_reg
      lr_reg = LogisticRegression()
      lr_reg.fit(X_train,y_train)
      logger.info(f'LogisticRegression Test Accuracy : {accuracy_score(y_test,lr_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def dt(X_train,y_train,X_test,y_test):
    try:
      global dt_reg
      dt_reg = DecisionTreeClassifier(criterion='entropy')
      dt_reg.fit(X_train,y_train)
      logger.info(f'DecisionTreeClassifier Test Accuracy : {accuracy_score(y_test,dt_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def rf(X_train,y_train,X_test,y_test):
    try:
      global rf_reg
      rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
      rf_reg.fit(X_train,y_train)
      logger.info(f'RandomForestClassifier Test Accuracy : {accuracy_score(y_test,rf_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def svm(X_train, y_train, X_test, y_test):
    try:
        global svm_reg
        svm_reg = SVC(probability=True)  # probability=True for predict_proba
        svm_reg.fit(X_train, y_train)
        logger.info(f'SVM Test Accuracy : {accuracy_score(y_test, svm_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def gbc(X_train, y_train, X_test, y_test):
    try:
        global gbc_reg
        gbc_reg = GradientBoostingClassifier()
        gbc_reg.fit(X_train, y_train)
        logger.info(f'GradientBoostingClassifier Test Accuracy : {accuracy_score(y_test, gbc_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def ada(X_train, y_train, X_test, y_test):
    try:
        global ada_reg
        ada_reg = AdaBoostClassifier()
        ada_reg.fit(X_train, y_train)
        logger.info(f'AdaBoostClassifier Test Accuracy : {accuracy_score(y_test, ada_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def etc(X_train, y_train, X_test, y_test):
    try:
        global etc_reg
        etc_reg = ExtraTreesClassifier()
        etc_reg.fit(X_train, y_train)
        logger.info(f'ExtraTreesClassifier Test Accuracy : {accuracy_score(y_test, etc_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def xgb_model(X_train, y_train, X_test, y_test):
    try:
        global xgb_reg
        xgb_reg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_reg.fit(X_train, y_train)
        logger.info(f'XGBoost Test Accuracy : {accuracy_score(y_test, xgb_reg.predict(X_test))}')
    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


def common(X_train, y_train, X_test, y_test):
    try:
        logger.info('=========KNN===========')
        knn(X_train, y_train, X_test, y_test)
        logger.info('=========NB===========')
        nb(X_train, y_train, X_test, y_test)
        logger.info('=========LR===========')
        lr(X_train, y_train, X_test, y_test)
        logger.info('=========DT===========')
        dt(X_train, y_train, X_test, y_test)
        logger.info('=========RF===========')
        rf(X_train, y_train, X_test, y_test)

        logger.info('=========SVM===========')
        svm(X_train, y_train, X_test, y_test)
        logger.info('=========Gradient Boosting===========')
        gbc(X_train, y_train, X_test, y_test)
        logger.info('=========AdaBoost===========')
        ada(X_train, y_train, X_test, y_test)
        logger.info('=========Extra Trees===========')
        etc(X_train, y_train, X_test, y_test)
        logger.info('=========XGBoost===========')
        xgb_model(X_train, y_train, X_test, y_test)

        # Collect predictions for ROC curve
        knn_predictions = knn_reg.predict_proba(X_test)[:, 1]
        naive_predictions = naive_reg.predict_proba(X_test)[:, 1]
        lr_predictions = lr_reg.predict_proba(X_test)[:, 1]
        dt_predictions = dt_reg.predict_proba(X_test)[:, 1]
        rf_predictions = rf_reg.predict_proba(X_test)[:, 1]
        svm_predictions = svm_reg.predict_proba(X_test)[:, 1]
        gbc_predictions = gbc_reg.predict_proba(X_test)[:, 1]
        ada_predictions = ada_reg.predict_proba(X_test)[:, 1]
        etc_predictions = etc_reg.predict_proba(X_test)[:, 1]
        xgb_predictions = xgb_reg.predict_proba(X_test)[:, 1]

        # plt.figure(figsize=(8, 5))
        # plt.plot([0, 1], [0, 1], "k--")
        model_preds = {
            "KNN": knn_predictions,
            "NB": naive_predictions,
            "LR":   lr_predictions,
            "DT": dt_predictions,
            "RF": rf_predictions,
            "SVM": svm_predictions,
            "GBC": gbc_predictions,
            "ADA": ada_predictions,
            "ETC": etc_predictions,
            "XGA": xgb_predictions
        }
        # plt.plot(*roc_curve(y_test, knn_predictions)[:2], label="KNN")
        # plt.plot(*roc_curve(y_test, naive_predictions)[:2], label="NB")
        # plt.plot(*roc_curve(y_test, lr_predictions)[:2], label="LR")
        # plt.plot(*roc_curve(y_test, dt_predictions)[:2], label="DT")
        # plt.plot(*roc_curve(y_test, rf_predictions)[:2], label="RF")
        # plt.plot(*roc_curve(y_test, svm_predictions)[:2], label="SVM")
        # plt.plot(*roc_curve(y_test, gbc_predictions)[:2], label="Gradient Boosting")
        # plt.plot(*roc_curve(y_test, ada_predictions)[:2], label="AdaBoost")
        # plt.plot(*roc_curve(y_test, etc_predictions)[:2], label="Extra Trees")
        # plt.plot(*roc_curve(y_test, xgb_predictions)[:2], label="XGBoost")

        auc_scores = {}
        for name, pred in model_preds.items():
            fpr, tpr, _ = roc_curve(y_test, pred)
            plt.plot(fpr, tpr, label=name)
            auc_scores[name] = roc_auc_score(y_test, pred)

        # plt.xlabel("FPR")
        # plt.ylabel("TPR")
        # plt.title("ROC Curve - ALL Models")
        # plt.legend(loc='lower right')
        # plt.show()

        # ------------------ ROC-AUC SCORE CALCULATION ------------------
        for model, score in auc_scores.items():
            logger.info(f"{model} ROC-AUC Score: {score}")

            # ------------------ SELECT BEST MODEL ------------------
        best_model_name = max(auc_scores, key=auc_scores.get)
        best_auc = auc_scores[best_model_name]


        logger.info("===================================")
        logger.info(f"BEST MODEL: {best_model_name}")
        logger.info(f"BEST ROC-AUC: {best_auc}")
        logger.info("===================================")

        # Map model name to object
        model_dict = {
            "KNN": knn,
            "NB": nb,
            "LR": lr,
            "DT": dt,
            "RF": rf,
            "SVM": svm,
            "GBC": gbc,
            "ADA": ada,
            "ETC": etc,
            "XGA": xgb_model
        }

        best_model = model_dict[best_model_name]

        # ------------------ SAVE BEST MODEL ------------------
        with open("churn_prediction.pkl", "wb") as f:
            pickle.dump(eval(best_model_name.lower() + "_reg"), f)
        logger.info(f"{best_model_name} saved successfully as churn_Prediction.pkl")


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
