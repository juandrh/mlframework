# Importing core libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing

"""
    Generate new polynomial features
    Encode with One Hot and Ordinal Encoding 

"""

def process_features(df,cat_columns, num_columns, create_new = True):  # if create_new = True --> only Ordinal encoding

    if create_new:
        print("Creating new features", end=" - ")
    else:
        print("Encoding features", end=" - ")

    if  create_new:

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

    if  create_new:
        #  join dataframes
        final_df = pd.concat([df, OH_cols_df], axis=1)        
    else:
        final_df =  df.copy()   

    # change columns names (because of the generated 'int' columns with OHE)
    useful_features = [c for c in final_df.columns if c not in ("id", "target", "kfold")] 
    for c in useful_features:
        final_df["_"+str(c)]=final_df[c] 

    print("Done!", end=" - ")
    return final_df

