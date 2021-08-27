import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import joblib
import optuna
from . import feature_generator

"""
    hyperparameters optimizer
"""


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))

def run(trial):
    if MODEL == 5:
        print("\nXGBRegressor")
        # XGBRegressor parameter gaps 
        n_estimators = trial.suggest_int("n_estimators", 1000, 8000)
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
        max_depth = trial.suggest_int("max_depth", 1, 7)
    if MODEL == 6:
        print("\nLGBMRegressor")
        # XGBRegressor parameter gaps 
        n_estimators = trial.suggest_int("n_estimators", 1000, 8000)
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
        max_depth = trial.suggest_int("max_depth", 1, 256)
        num_leaves= trial.suggest_int("num_leaves",1,100)
            

    scores=[]
    for fold in range(FOLD):        
        xtrain = new_df[new_df.kfold != fold].reset_index(drop=True)
        xvalid = new_df[new_df.kfold == fold].reset_index(drop=True)  
        ytrain = xtrain.target
        yvalid = xvalid.target    
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]
   
        # standarization

        scaler = preprocessing.StandardScaler()
        xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])
        xvalid[numerical_cols] = scaler.transform(xvalid[numerical_cols])   

        # data is ready to train

        print(fold,end=" - ")

        # change model selecction by hand 
        # 
        if MODEL == 5:            
            model = XGBRegressor(random_state=42,n_jobs=-1,tree_method='gpu_hist',
                            eval_metric='rmse',predictor='gpu_predictor',
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            subsample= subsample,
                            max_depth=max_depth,
                            colsample_bytree= colsample_bytree,
                            reg_lambda = reg_lambda,
                            reg_alpha = reg_alpha,                 
                            objective='reg:squarederror')
        if MODEL == 6:            
            model = lgb.LGBMRegressor(random_state=42,n_jobs=-1,
                            metrics ='rmse',
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            subsample= subsample,
                            max_depth=max_depth,
                            colsample_bytree= colsample_bytree,
                            reg_lambda = reg_lambda,
                            reg_alpha = reg_alpha,          
                            num_leaves = num_leaves,
                            )

        model.fit(xtrain, ytrain)  
        preds_valid = model.predict(xvalid)
        rmse = mean_squared_error(yvalid, preds_valid, squared=False)
        print(rmse)
        scores.append(rmse)

    return np.mean(scores)


if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA) 
    print("Data loaded")

    useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
    object_cols = [col for col in useful_features if 'cat' in col]
    numerical_cols = [col for col in useful_features if 'cont' in col]

    # drop outliers from targer colummn
    df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")
    print("Num. folds: ",FOLD)

    # process features
    new_df=feature_generator.process_features(df,object_cols,numerical_cols,False)
    useful_features = [c for c in new_df.columns if (c not in ("id", "target", "kfold") and str(c).startswith('_'))]
    numerical_cols = [col for col in useful_features if str(col).startswith('_cont')]
   
    # Start study 
    study = optuna.create_study(direction="minimize")
    study.optimize(run, n_trials=30)

    print("\n")
    print(study.best_params)

    # save to file best parameters found
    joblib.dump(study.best_params, f"models/model{MODEL}__{FOLD}_best_params.pkl")

    print("Best params saved")
    
 
