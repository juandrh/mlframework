
import numpy as np
import pandas as pd
import os, warnings, random, time, pickle
from joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

TEST_DATA = os.environ.get("TEST_DATA")
TRAINING_DATA = os.environ.get("TRAINING_DATA")


## Memory Reducer adapted from Konstantin Yakovlev (https://www.kaggle.com/kyakovlev/ieee-small-tricks)
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    print(df.info())   
    
    start_mem = df.memory_usage().sum() / 1024**2     
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()        
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2    
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    print(df.info())
    return df



if __name__ == "__main__":

    # testing Memory reducer
    df = pd.read_csv(TRAINING_DATA)     
    df = reduce_mem_usage(df, verbose=True)

    
   

  
    
    