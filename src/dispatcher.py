from sklearn import ensemble
from xgboost import XGBRegressor


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