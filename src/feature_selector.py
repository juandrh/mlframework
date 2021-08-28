import numpy as np
import pandas as pd
import os
import joblib
from BorutaShap import BorutaShap
from sklearn import preprocessing
from . import dispatcher
from . import feature_generator

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


    """ df = df.drop(df[df['target'].lt(6)].index)
    print("Dropped ",300000-len(df), " target outliers")
    print("Num. folds: ",FOLD) """


    # standarization numerical features 
    scaler = preprocessing.StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # generate and process features
    final_df=feature_generator.process_features(df,object_cols,numerical_cols,False)

    useful_features = [c for c in final_df.columns if (c not in ("id", "target", "kfold") and str(c).startswith('_'))]       
    print("Usefull features= ",useful_features)

    selected_columns = list()    

    for fold in range(1):   
        xtrain =  final_df[final_df.kfold != fold].reset_index(drop=True)

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

    # save to file
    joblib.dump(final_selection, f"models/model{MODEL}_{FOLD}_features.pkl")

    print(final_selection)


  # model 4 descartaer -->'_cat4', '_cat6', '_cat9', '_cat7', '_cat2'
  

