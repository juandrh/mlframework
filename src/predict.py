import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    #predictions = None
    final_predictions = []

    for fold in range(FOLD):
        df = pd.read_csv(test_data_path)
        encoders = joblib.load(os.path.join(model_path, f"{model_type}_{fold}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{fold}_{FOLD}_columns.pkl"))

        
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{fold}_{FOLD}_.pkl"))

        useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
        object_cols = [col for col in useful_features if 'cat' in col]
        numerical_cols = [col for col in useful_features if 'cont' in col]

        df = df[useful_features]
        xtest = df_test.copy()

        # standarization

        scaler = preprocessing.StandardScaler()
        xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])

        # categorical features
        high_cardinality_cols = [col for col in object_cols if xtest[col].nunique()>=9]
        low_cardinality_cols = [col for col in object_cols if xtest[col].nunique()<9]
        
        
        # label encode columns with high cardinality 
        ordinal_encoder = preprocessing.OrdinalEncoder()
        xtest[high_cardinality_cols] = ordinal_encoder.fit_transform(xtest[high_cardinality_cols])
    
        # One hot encode columns with low cardinality 
        OH_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
        OH_cols_test = pd.DataFrame(OH_encoder.transform(xtest[low_cardinality_cols]))

        # codificador one-hot elimina; ponerlo de nuevo 
        OH_cols_test.index = xtest.index

        # Eliminar columnas categóricas (se reemplazarán con codificación one-hot) 
        num_X_test= xtest.drop(low_cardinality_cols, axis=1)

        #  añadir columnas codificadas one-hot a variables numéricas 
        after_OH_test= pd.concat([num_X_test, OH_cols_test], axis=1) 


        test_preds = clf.predict(after_OH_test)
        final_predictions.append(test_preds)
          
    sub = np.mean(np.column_stack(final_predictions), axis=1)  
    return sub
    

if __name__ == "__main__":


    submission = predict(test_data_path=TEST_DATA, 
                         model_type= MODEL, 
                         model_path="models/")

    sample_submission.target = submission
    sample_submission.to_csv(f"models/model{MODEL}_submission.csv", index=False)   


