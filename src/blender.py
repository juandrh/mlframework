import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLDS = int(os.environ.get("FOLDS"))
MODEL = int(os.environ.get("MODEL"))

if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    sample_submission = pd.read_csv("input/sample_submission.csv")
    print("Data loaded")

    # Deleting outliers from target column
    df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")

    dfs = []
    dfs_test = []
    useful_features = []
    
    for model in range(8):
        dfs.append(pd.read_csv(f"output/model{model}_{FOLDS}_train_pred.csv"))
        dfs_test.append(pd.read_csv(f"output/model{model}_{FOLDS}_test_pred.csv"))
        df = df.merge(dfs[model], on="id", how="left")
        df_test = df_test.merge(dfs_test[model], on="id", how="left")
        useful_features.append(f"pred_{model}")
        print(f"model {model} blended")

    df_test = df_test[useful_features]

    print(df.shape,df_test.shape)  

    final_predictions = []
    scores = []
    for fold in range(FOLDS):
        xtrain =  df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()

        ytrain = xtrain.target
        yvalid = xvalid.target
        
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

        xtrain = xtrain.fillna(0)
        xvalid = xvalid.fillna(0)   
  
        model = LinearRegression()
        # model = XGBRegressor(random_state=42, 
        #                  n_jobs=-1,
        #                  n_estimators= 1000,
        #                  tree_method='gpu_hist',
        #                  learning_rate= 0.08970028112557221,
        #                  subsample= 0.9487438254800091,
        #                  max_depth= 2,
        #                  colsample_bytree= 0.3685425845467418,
        #                  reg_lambda = 9.309499343828611e-07,
        #                  reg_alpha = 23.955318691526553,
        #                  eval_metric='rmse',
        #                  predictor='gpu_predictor',
        #                  objective='reg:squarederror')
        model.fit(xtrain, ytrain)
        
        preds_valid = model.predict(xvalid)
        test_preds = model.predict(xtest)
        final_predictions.append(test_preds)
        rmse = mean_squared_error(yvalid, preds_valid, squared=False)
        print(fold, rmse)
        scores.append(rmse)

    print(np.mean(scores), np.std(scores))

    sample_submission.target = np.mean(np.column_stack(final_predictions), axis=1)
    sample_submission.to_csv("submission_blended.csv", index=False)
    print("Submission from blended models saved")
    print(model.coef_)

