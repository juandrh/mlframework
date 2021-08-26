import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor, DMatrix
import joblib

import matplotlib.pyplot as plt
            
import seaborn as sns
from time import time
import pprint
import joblib
from functools import partial
# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Model selection
from sklearn.model_selection import KFold

import optuna

from . import dispatcher



"""
    hyperparameters optimizer
"""


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))

def run(trial):

    n_estimators = trial.suggest_int("n_estimators", 1000, 8000)
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)

    final_predictions = []
    scores=[]
    for fold in range(FOLD):
        xtrain =  df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()

        ytrain = xtrain.target
        yvalid = xvalid.target
        
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

   
        # standarization

        scaler = preprocessing.StandardScaler()
        xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])
        xvalid[numerical_cols] = scaler.transform(xvalid[numerical_cols])
        xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])

        # categorical features
        high_cardinality_cols = [col for col in object_cols if xtrain[col].nunique()>=9]
        low_cardinality_cols = [col for col in object_cols if xtrain[col].nunique()<9]
        
        
        # label encode columns with high cardinality 
        ordinal_encoder = preprocessing.OrdinalEncoder()
        xtrain[high_cardinality_cols] = ordinal_encoder.fit_transform(xtrain[high_cardinality_cols])
        xvalid[high_cardinality_cols] = ordinal_encoder.fit_transform(xvalid[high_cardinality_cols])
        xtest[high_cardinality_cols] = ordinal_encoder.fit_transform(xtest[high_cardinality_cols])
    
        # One hot encode columns with low cardinality 
        OH_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)

        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(xtrain[low_cardinality_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(xvalid[low_cardinality_cols]))
        OH_cols_test = pd.DataFrame(OH_encoder.transform(xtest[low_cardinality_cols]))

        # codificador one-hot elimina; ponerlo de nuevo 
        OH_cols_train.index = xtrain.index
        OH_cols_valid.index = xvalid.index
        OH_cols_test.index = xtest.index

        # Eliminar columnas categóricas (se reemplazarán con codificación one-hot) 
        num_X_train = xtrain.drop(low_cardinality_cols, axis=1)
        num_X_valid = xvalid.drop(low_cardinality_cols, axis=1)
        num_X_test= xtest.drop(low_cardinality_cols, axis=1)

        #  añadir columnas codificadas one-hot a variables numéricas 
        after_OH_xtrain = pd.concat([num_X_train, OH_cols_train], axis=1)
        after_OH_valid= pd.concat([num_X_valid, OH_cols_valid], axis=1)
        after_OH_test= pd.concat([num_X_test, OH_cols_test], axis=1) 

        # data is ready to train

        print(fold,end=" - ")
        clf = dispatcher.MODELS[MODEL]
        clf.fit(after_OH_xtrain, ytrain)
        
        #print(metrics.roc_auc_score(yvalid, preds))

        preds_valid = clf.predict(after_OH_valid)
        test_preds = clf.predict(after_OH_test)
        final_predictions.append(test_preds)
        rmse = mean_squared_error(yvalid, preds_valid, squared=False)
        print(rmse)
        scores.append(rmse)

    return np.mean(scores)
change

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    print("Data loaded")
    useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
    object_cols = [col for col in useful_features if 'cat' in col]
    numerical_cols = [col for col in useful_features if 'cont' in col]
    df_test = df_test[useful_features]

    df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")
    print("Num. folds: ",FOLD)

    label_encoders = {}
    """ for c in object_cols:
        lbl = preprocessing.LabelEncoder()
        df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")        
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(df[c].values.tolist() +
                df_test[c].values.tolist())
        df.loc[:, c] = lbl.transform(df[c].values.tolist())        
        label_encoders[c] = lbl """


    study = optuna.create_study(direction="minimize")
    study.optimize(run, n_trials=1)

    print("\n")
    print(study.best_params)

    joblib.dump(study.best_params, f"models/model{MODEL}__{FOLD}_best_params.pkl")

    print("Best params saved")
    
 
