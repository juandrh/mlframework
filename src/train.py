import os
import pandas as pd
import numpy as np
import time
#from sklearn import ensemble
from sklearn import preprocessing
#from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error
from . import dispatcher
from . import feature_generator
from . import utils

"""
    For training model using Cross Validation
"""

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLDS = int(os.environ.get("FOLDS"))
MODEL = int(os.environ.get("MODEL"))

if __name__ == "__main__":

    start_time=time.perf_counter()
    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    sample_submission = pd.read_csv("input/sample_submission.csv")
    print("Data loaded")



    useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
    object_cols = [col for col in useful_features if 'cat' in col]
    numerical_cols = [col for col in useful_features if 'cont' in col]
    df_test = df_test[useful_features]

    # Deleting outliers from target column
    df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")
    print("Num. folds: ",FOLDS)
    print("Model: ",str(dispatcher.MODELS[MODEL]).split("(")[0])

    # process features
    new_df=feature_generator.process_features(df,object_cols,numerical_cols,False)
    new_df_test=feature_generator.process_features(df_test,object_cols,numerical_cols,False)


    useful_features = [c for c in new_df.columns if (c not in ("id", "target", "kfold") and str(c).startswith('_'))]
    numerical_cols = [col for col in useful_features if str(col).startswith('_cont')]
    new_df_test = new_df_test[useful_features]    
   
    final_test_predictions = []
    final_valid_predictions = {}
    scores=[]
    print("\n")
    

    # cross validation loop
    for fold in range(FOLDS):
        xtrain =  new_df[new_df.kfold != fold].reset_index(drop=True)
        xvalid = new_df[new_df.kfold == fold].reset_index(drop=True)
        xtest = new_df_test.copy()
        valid_ids = xvalid.id.values.tolist()

        ytrain = xtrain.target
        yvalid = xvalid.target
        
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

            #prueba --------------------------
        xtrain = utils.reduce_mem_usage(xtrain, verbose=True)
        xvalid = utils.reduce_mem_usage(xvalid, verbose=True)
   
        # standarization

        scaler = preprocessing.StandardScaler()
        xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])
        xvalid[numerical_cols] = scaler.transform(xvalid[numerical_cols])
        xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])
    
        # data is ready to train
        print(fold,end=" - ")
        model = dispatcher.MODELS[MODEL]
        model.fit(xtrain, ytrain)
        
        preds_valid = model.predict(xvalid)
        test_preds = model.predict(xtest)  
        rmse = mean_squared_error(yvalid, preds_valid, squared=False)
        print(rmse)
        scores.append(rmse)
        final_test_predictions.append(test_preds)
        final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))

        # save model to file
        joblib.dump(model, f"models/model{MODEL}_{fold}_{FOLDS}_.pkl")        
    
   
    print(f"Time elapsed {round(time.perf_counter()-start_time,2)} s.")
    print (np.mean(scores),np.std(scores))
    final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
    final_valid_predictions.columns = ["id", f"pred_{MODEL}"]
    final_valid_predictions.to_csv(f"output/model{MODEL}_{FOLDS}_train_pred.csv", index=False)

    sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
    sample_submission.columns = ["id", f"pred_{MODEL}"]
    sample_submission.to_csv(f"output/model{MODEL}_{FOLDS}_test_pred.csv", index=False)



    


