from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor

import joblib
import os

FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))


MODELS = {
    0: LinearRegression (n_jobs=-1),
    1: SVR(kernel='rbf', gamma='auto',degree=3,C =1.0),
    2: DecisionTreeRegressor(
                        criterion='mse',
                        random_state=42,
                        splitter='best'),
    3: ensemble.RandomForestRegressor(n_estimators = 300,max_depth=2, n_jobs=-1 ,random_state = 42),
    4: ensemble.GradientBoostingRegressor(random_state=42,learning_rate=1.0,n_estimators=50,),
    5: XGBRegressor(random_state=42, 
                         n_jobs=-1,
                         n_estimators= 5234,
                         tree_method='gpu_hist',
                         learning_rate= 0.08970028112557221,
                         subsample= 0.9487438254800091,
                         max_depth= 2,
                         colsample_bytree= 0.3685425845467418,
                         reg_lambda = 9.309499343828611e-07,
                         reg_alpha = 23.955318691526553,
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
        
        best_params = joblib.load(os.path.join(f"models/model{MODEL}__{FOLD}_best_params.pkl"))

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
  