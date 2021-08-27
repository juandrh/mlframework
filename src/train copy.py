import os
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))

print("Libs imported")
if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)
    print("data loaded")
    useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
    object_cols = [col for col in useful_features if 'cat' in col]
    numerical_cols = [col for col in useful_features if 'cont' in col]
    df_test = df_test[useful_features]

    df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")
    print("Num. folds: ",FOLD)

    final_predictions = []
    scores=[]
    for fold in range(FOLD):
        xtrain =  df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()

        ytrain = xtrain.target
        yvalid = xvalid.target
        
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]

   
        # standarization

        scaler = preprocessing.StandardScaler()
        xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])
        xvalid[numerical_cols] = scaler.transform(xvalid[numerical_cols])
        xtest[numerical_cols] = scaler.transform(xtest[numerical_cols])

        # categorical features
        high_cardinality_cols = [col for col in object_cols if xtrain[col].nunique()>=9]
        low_cardinality_cols = [col for col in object_cols if xtrain[col].nunique()<9]
        
        
        # label encode columns with high cardinality 
        ordinal_encoder = preprocessing.OrdinalEncoder()
        xtrain[high_cardinality_cols] = ordinal_encoder.fit_transform(xtrain[high_cardinality_cols])
        xvalid[high_cardinality_cols] = ordinal_encoder.fit_transform(xvalid[high_cardinality_cols])
        xtest[high_cardinality_cols] = ordinal_encoder.fit_transform(xtest[high_cardinality_cols])
    
        # One hot encode columns with low cardinality 
        OH_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)

        OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(xtrain[low_cardinality_cols]))
        OH_cols_valid = pd.DataFrame(OH_encoder.transform(xvalid[low_cardinality_cols]))
        OH_cols_test = pd.DataFrame(OH_encoder.transform(xtest[low_cardinality_cols]))

        # codificador one-hot elimina; ponerlo de nuevo 
        OH_cols_train.index = xtrain.index
        OH_cols_valid.index = xvalid.index
        OH_cols_test.index = xtest.index

        # Eliminar columnas categóricas (se reemplazarán con codificación one-hot) 
        num_X_train = xtrain.drop(low_cardinality_cols, axis=1)
        num_X_valid = xvalid.drop(low_cardinality_cols, axis=1)
        num_X_test= xtest.drop(low_cardinality_cols, axis=1)

        #  añadir columnas codificadas one-hot a variables numéricas 
        after_OH_xtrain = pd.concat([num_X_train, OH_cols_train], axis=1)
        after_OH_valid= pd.concat([num_X_valid, OH_cols_valid], axis=1)
        after_OH_test= pd.concat([num_X_test, OH_cols_test], axis=1) 

  

    
        # data is ready to train

        print(fold,end=" - ")
        model = dispatcher.MODELS[MODEL]
        model.fit(after_OH_xtrain, ytrain)
        
        #print(metrics.roc_auc_score(yvalid, preds))

        preds_valid = model.predict(after_OH_valid)
        test_preds = model.predict(after_OH_test)
        final_predictions.append(test_preds)
        rmse = mean_squared_error(yvalid, preds_valid, squared=False)
        print(rmse)
        scores.append(rmse)

        joblib.dump(model, f"models/model{MODEL}_{fold}_{FOLD}_.pkl")
        joblib.dump(after_OH_xtrain.columns, f"models/model{MODEL}_{fold}_{FOLD}_columns.pkl")
    
    print (np.mean(scores),np.std(scores))



    


