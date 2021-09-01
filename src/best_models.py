import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import lazypredict
from lazypredict import Supervised
from lazypredict.Supervised import LazyRegressor
from sklearn.preprocessing import OrdinalEncoder


"""
    Using LazyRegressor for searching best models with given data
    This is a baseline aproximation

    Author:  Sanskar Hasija (https://www.kaggle.com/odins0n/30dml-comparison-of-36-different-models)

"""


if __name__ == "__main__":


    plt.style.use('fivethirtyeight')
    plt.rcParams["figure.figsize"] = (20,5)
    num_models = 36   #Number of Models

    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")

    cat_features = ["cat" + str(i) for i in range(10)]
    num_features = ["cont" + str(i) for i in range(14)]

    for col in cat_features:
        encoder = OrdinalEncoder()
        train[col] = encoder.fit_transform(np.array(train[col]).reshape(-1, 1))
        test[col] = encoder.transform(np.array(test[col]).reshape(-1, 1))
        
    X = train.drop(["id", "target"], axis=1)
    X_test = test.drop(["id"], axis=1)
    y = train["target"]


    #Spliting into training and validation set
    offset = int(X.shape[0] * 0.67)
    X_train, y_train = X[:offset], y[:offset]
    X_valid, y_valid = X[offset:], y[offset:]

    reg_idx = [i for i in range(num_models)]
    noregs_idx = [10,15,23,24,29,32] # Removing 6 models from 42 models. Some of these models are time consuming whereas other require lot of ram.
    regs_name =[]
    regs = []
    for i in range(42):
        regs_name.append(lazypredict.Supervised.REGRESSORS[i][0])
        regs.append(lazypredict.Supervised.REGRESSORS[i][1])

    for i in noregs_idx:
        del regs_name[i]
        del regs[i]

    

    print("ALL 36 AVAILABLE REGRESSION MODELS:")
    for i in range(num_models):
        print(i+1 , regs_name[i])

    results = pd.DataFrame()


    for i in range(num_models):
        reg = LazyRegressor(verbose=0, 
                        ignore_warnings=False,
                        custom_metric=None,
                        regressors = [regs[i]])
        models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)
        models.index = [regs_name[i]]
        results = results.append(models)
    

    results = results.sort_values(by = "RMSE")
    print(results)



