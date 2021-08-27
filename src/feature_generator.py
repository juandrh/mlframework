# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial
import os
# Feature selection
from BorutaShap import BorutaShap
# Data processing
from sklearn import preprocessing
from . import dispatcher


def generate_features(df,cat_columns, num_columns):

    # NUMERICAL FEATURES
    # polynomial features
    poly = preprocessing.PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    data_poly = poly.fit_transform(df[num_columns])    
    df_poly = pd.DataFrame(data_poly, columns= [f"poly_{i}" for i in range(data_poly.shape[1])])   
    df = pd.concat([df, df_poly], axis = 1)
   
    # CATEGORICAL FEATURES
    #  one hot encoding 
    OH_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_df = pd.DataFrame(OH_encoder.fit_transform(df[cat_columns]))

    # put index back
    OH_cols_df.index = df.index

    # label encoding
    ordinal_encoder = preprocessing.OrdinalEncoder()
    df[cat_columns] = ordinal_encoder.fit_transform(df[cat_columns])

    #  join dataframes
    after_OH_df = pd.concat([df, OH_cols_df], axis=1)

    useful_features = [c for c in after_OH_df.columns if c not in ("id", "target", "kfold")]   
    
    # change columns names (because of the generate 'int' columns)
    for c in useful_features:
        after_OH_df["n_"+str(c)]=after_OH_df[c]
     
    useful_features = [c for c in after_OH_df.columns if str(c).startswith('n_')]



    return useful_features

