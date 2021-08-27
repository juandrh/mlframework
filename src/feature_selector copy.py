# Importing core libraries
import numpy as np
import pandas as pd
#import pprint
#from functools import partial
import os

# Classifier/Regressor
from xgboost import XGBRegressor

# Feature selection
from BorutaShap import BorutaShap

# Data processing
from sklearn import preprocessing

# Feature selection
from BorutaShap import BorutaShap

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = int(os.environ.get("MODEL"))

print("Libs imported")
if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)

    print("Data loaded")
    useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
    object_cols = [col for col in useful_features if 'cat' in col]
    numerical_cols = [col for col in useful_features if 'cont' in col]


    df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")
    print("Num. folds: ",FOLD)

    
    """ 
    label_encoders = {}
    for c in object_cols:
        lbl = preprocessing.LabelEncoder()
        df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")        
        df_test.loc[:, c] = df_test.loc[:, c].astype(str).fillna("NONE")
        lbl.fit(df[c].values.tolist() +
                df_test[c].values.tolist())
        df.loc[:, c] = lbl.transform(df[c].values.tolist())        
        label_encoders[c] = lbl """

    # standarization

    scaler = preprocessing.StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # One hot encode categorical columns
    OH_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(df[object_cols]))

    # codificador one-hot elimina; ponerlo de nuevo 
    OH_cols_train.index = df.index

    # label encode categorical columns
    ordinal_encoder = preprocessing.OrdinalEncoder()
    df[object_cols] = ordinal_encoder.fit_transform(df[object_cols])

    #  añadir columnas codificadas one-hot a variables numéricas 
    after_OH_df = pd.concat([df, OH_cols_train], axis=1)

    useful_features = [c for c in after_OH_df.columns if c not in ("id", "target", "kfold")]   
    
    for c in useful_features:
        after_OH_df["n_"+str(c)]=after_OH_df[c]
     
    useful_features = [c for c in after_OH_df.columns if str(c).startswith('n_')]

    selected_columns = list()

    

    for fold in range(FOLD):
        xtrain =  after_OH_df[after_OH_df.kfold != fold].reset_index(drop=True)

        ytrain = xtrain.target        
        xtrain = xtrain[useful_features]     
           
        # data is ready to train
        print(fold," / ",FOLD)

        model = dispatcher.MODELS[MODEL]            
      
        Feature_Selector = BorutaShap(model=model,
                                  importance_measure='shap', 
                                  classification=False)


        Feature_Selector.fit(X=xtrain, y=ytrain, n_trials=50, random_state=0)

    
        #Feature_Selector.plot(which_features='all', figsize=(24,12))
    
        selected_columns.append(sorted(Feature_Selector.Subset().columns))

        print(f"Selected features at fold {fold} are: {selected_columns[-1]}")
        
 
    final_selection = sorted({item for selection in selected_columns for item in selection})
    print(final_selection)


    
  

