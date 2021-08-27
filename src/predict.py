import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
import joblib
import numpy as np

from . import dispatcher
from . import feature_generator

TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))

"""
    Generates predictions using a trained and saved model

"""


def predict(test_data_path, model_type, model_path):

    final_predictions = []   
        
    for fold in range(FOLD):
        df = pd.read_csv(test_data_path)        

        #cols = joblib.load(os.path.join(model_path, f"model{model_type}_{fold}_{FOLD}_columns.pkl"))        
        model = joblib.load(os.path.join(model_path, f"model{model_type}_{fold}_{FOLD}_.pkl"))
        print(f"Model {fold}_{FOLD} loaded", end=" - ")

        useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
        object_cols = [col for col in useful_features if 'cat' in col]
        numerical_cols = [col for col in useful_features if 'cont' in col]

        # process features
        new_df=feature_generator.process_features(df,object_cols,numerical_cols,False)
        useful_features = [c for c in new_df.columns if (c not in ("id", "target", "kfold") and str(c).startswith('_'))]
        numerical_cols = [col for col in useful_features if str(col).startswith('_cont')]

        xtest = new_df[useful_features].copy()
        

        # standarization
        scaler = preprocessing.StandardScaler()
        xtest[numerical_cols] = scaler.fit_transform(xtest[numerical_cols])
        
        test_preds = model.predict(xtest)
        final_predictions.append(test_preds)
        print("Prediction done.")
          
    sub = np.mean(np.column_stack(final_predictions), axis=1)  
    print(sub)
    return sub
    

if __name__ == "__main__":

    print("Prediction init")
    sample_submission = pd.read_csv("input/sample_submission.csv")

    submission = predict(test_data_path=TEST_DATA, 
                         model_type= MODEL, 
                         model_path="models/")
    
    sample_submission.target = submission
    sample_submission.to_csv(f"models/model{MODEL}_submission.csv", index=False)   
    print("Submission saved")

