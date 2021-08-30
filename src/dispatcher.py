from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn import linear_model
from sklearn import ensemble
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

import joblib
import os

FOLDS = int(os.environ.get("FOLDS"))
MODEL = int(os.environ.get("MODEL"))


MODELS = {
    0: LinearRegression (n_jobs=-1),
    1: HistGradientBoostingRegressor(
                        loss='least_squares',
                        learning_rate=0.1, 
                        max_iter=100,
                        max_leaf_nodes=31, 
                        l2_regularization=0.0, 
                        max_bins=255, 
                        verbose=0,
                        random_state=42),
    2: linear_model.BayesianRidge(
                       n_iter=300,
                        tol=0.001, 
                        alpha_1=1e-06, 
                        alpha_2=1e-06, 
                        lambda_1=1e-06,
                         lambda_2=1e-06),
    3: ensemble.ExtraTreesRegressor(
                        n_estimators=100,
                        criterion='mse',
                        max_depth=None, 
                        min_samples_split=2,
                        min_samples_leaf=1,
                        min_weight_fraction_leaf=0.0,         
                        n_jobs=-1, random_state=42, verbose=0),    
    4: ensemble.GradientBoostingRegressor(
                        random_state=42,
                        learning_rate=0.06,
                        n_estimators=36),
    5: XGBRegressor(random_state=42, 
                         n_jobs=-1,
                         n_estimators= 7318,
                         tree_method='gpu_hist',
                         learning_rate= 0.053608658184796765,
                         subsample= 0.8159790703280856,
                         max_depth= 2,
                         colsample_bytree= 0.4162844630643207,
                         reg_lambda = 0.08884988748314745,
                         reg_alpha = 1.1954167064703073e-06,
                         eval_metric='rmse',
                         predictor='gpu_predictor',
                         objective='reg:squarederror'),

                         
    6: lgb.LGBMRegressor (boosting_type='gbdt',
                        metric='rmse',
                        n_jobs=-1, 
                        verbose=-1,
                        random_state=42,
                        n_estimators= 2683,
                        learning_rate= 0.010250629304555186,
                        num_leaves= 79,
                        max_depth= 256,
                        subsample= 0.7778732709684482,
                        subsample_freq= 9,
                        colsample_bytree= 0.35917838955653647,
                        reg_lambda= 2.943257012154159,
                        reg_alpha= 2.416846681288718                     
                        ) ,
    7:MLPRegressor(                    
                    alpha=1e-2,
                    hidden_layer_sizes=(150,100,50),
                    random_state=42,                        
                    max_iter = 100,
                    activation ='relu',
                    solver = 'sgd',                    
                    learning_rate ='adaptive',
                    ) 
}


BEST_MODELS = {}

# create a dict from the optimized models 
def get_best_models(num_models):
    for model in range(num_models):
        
        best_params = joblib.load(os.path.join(f"models/model{MODEL}__{FOLDS}_best_params.pkl"))

        # change best_params format
        model_name = str(MODELS[MODEL]).split("(")[0]          
        bp=str(best_params)
        bp=bp.replace("{","(")
        bp=bp.replace("}",")")
        bp=bp.replace(":","=")
        bp=bp.replace("'","")    
        model_name = model_name + bp
        
        # add to dict
        BEST_MODELS[model+1] = model_name

        return BEST_MODELS


if __name__ == "__main__":

   print(get_best_models(1))
  