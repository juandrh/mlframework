import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sklearn import preprocessing
from . import feature_generator
from . import utils

"""
    For training model after using AutoML Tpot 
"""

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLDS = int(os.environ.get("FOLDS"))
MODEL = int(os.environ.get("MODEL"))


if __name__ == "__main__":

    
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

        
        xtrain = xtrain[useful_features]


            #prueba --------------------------
        xtrain = utils.reduce_mem_usage(xtrain, verbose=True)

   
        # standarization

        scaler = preprocessing.StandardScaler()
        xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])
        xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])


        # Average CV score on the training set was: -0.5377099242795287
        exported_pipeline = RandomForestRegressor(bootstrap=True, 
                                    max_features=0.6500000000000001,
                                    min_samples_leaf=17,
                                    min_samples_split=17, 
                                    n_estimators=100)

        
       
        # data is ready to train
        print(fold,end=" - ")
        exported_pipeline.fit(xtrain, xtrain)
      
        test_preds = exported_pipeline.predict(xtest)
        final_test_predictions.append(test_preds)
       
    sub = np.mean(np.column_stack(final_test_predictions), axis=1)  
    print(sub)

    sample_submission.target = np.mean(np.column_stack(final_test_predictions), axis=1)
    sample_submission.columns = ["id", f"pred_{MODEL}"]
    sample_submission.to_csv(f"output/tpot_pred.csv", index=False)



    
