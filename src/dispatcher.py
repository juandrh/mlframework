from sklearn import ensemble
from xgboost import XGBRegressor
import joblib
import os

FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))


MODELS = {
    0: ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),  #"randomforest"
    1: XGBRegressor(random_state=42, 
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
    2: ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),   #"extratrees"
    3: "extratrees"
}


BEST_MODELS = {}

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
  