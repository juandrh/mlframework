from sklearn import ensemble
from xgboost import XGBRegressor

0.75091


xgb_params = {'n_estimators': 500,
              'learning_rate': 0.5,
              'subsample': 0.926,
              'colsample_bytree': 0.84,
              'max_depth': 2,
              'booster': 'gbtree', 
              'reg_lambda': 35.1,
              'reg_alpha': 34.9,
              'random_state': 42,
              'verbose':2,
              'n_jobs': -1}

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "randomforestR": ensemble.RandomForestRegressor(n_estimators=500, n_jobs=-1, verbose=2),
    "XGBRegressor": XGBRegressor(**xgb_params),
}