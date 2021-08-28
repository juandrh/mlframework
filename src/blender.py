import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from . import dispatcher


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))





if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    sample_submission = pd.read_csv("input/sample_submission.csv")
    print("Data loaded")

    dfs = []
    dfs_test = []
    useful_features = []
    for model,_ in dispatcher.MODELS:
        dfs.append(pd.read_csv(f"output/model{model}_{FOLD}_train_pred.csv"))
        dfs_test.append(pd.read_csv(f"output/model{model}_{FOLD}_test_pred.csv"))
        df = df.merge(dfs[model], on="id", how="left")
        df_test = df_test.merge(dfs_test[model], on="id", how="left")
        useful_features.append(f"pred_{model}")

    df_test = df_test[useful_features]

    print(df.head())
    print(df.shape)  
    

    final_predictions = []
    scores = []
    for fold in range(5):
        xtrain =  df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()

        ytrain = xtrain.target
        yvalid = xvalid.target
        
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]
        
        model = LinearRegression()
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

