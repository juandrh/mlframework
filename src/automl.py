
import pandas as pd
import os
from sklearn import preprocessing
import pandas as pd
from . import feature_generator
from . import utils

from tpot import TPOTRegressor


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLDS = int(os.environ.get("FOLDS"))
MODEL = int(os.environ.get("MODEL"))




if __name__ == "__main__":

    df = pd.read_csv(TRAINING_DATA)    
    print("Data loaded")

    useful_features = [c for c in df.columns if c not in ("id", "target", "kfold")]
    object_cols = [col for col in useful_features if 'cat' in col]
    numerical_cols = [col for col in useful_features if 'cont' in col]

    # drop outliers from targer colummn
    # df = df.drop(df[df['target'].lt(6)].index)
    # print("Dropped ",300000-len(df), " target outliers")
    # print("Num. folds: ",FOLD)

    

    # process features
    new_df=feature_generator.process_features(df,object_cols,numerical_cols,False)
   
    useful_features = [c for c in new_df.columns if (c not in ("id", "target", "kfold") and str(c).startswith('_'))]
    numerical_cols = [col for col in useful_features if str(col).startswith('_cont')]
  

    for fold in range(1):
            xtrain =  new_df[new_df.kfold == fold].reset_index(drop=True)
            ytrain = xtrain.target            
            xtrain = xtrain[useful_features]


                #prueba --------------------------
            xtrain = utils.reduce_mem_usage(xtrain, verbose=True)

    
            # standarization

            scaler = preprocessing.StandardScaler()
            xtrain[numerical_cols] = scaler.fit_transform(xtrain[numerical_cols])



            # create & fit TPOT classifier with 
            tpot = TPOTRegressor(generations=50, population_size=80, 
                                verbosity=2, early_stop=10,n_jobs=-1,scoring='neg_mean_squared_error')
            tpot.fit(xtrain, ytrain)

            # save our model code
            tpot.export('tpot_pipeline.py')



