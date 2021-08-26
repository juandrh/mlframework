import pandas as pd
from sklearn import model_selection
import sys
import os

""" 
    Split data into number of folds
    Default method Kfold (for StratifiedKFold uncomment it)
    Default number of folds = 5

"""

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))


if __name__ == "__main__":

    num_folds = FOLD    # default number of folds
    
    if len(sys.argv[:]) > 1:
        num_folds=int(sys.argv[1])
        print("Folds to create: ",sys.argv[1])
    else:
        print("Folds to create: ",FOLD)

    df = pd.read_csv(TRAINING_DATA)

    
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)


    # select split method
    kf = model_selection.KFold(n_splits=num_folds,shuffle=True,random_state=42)
    #kf = model_selection.StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
        print(fold, len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    file_name = "input/train_"+ str(num_folds)+"folds.csv"
    df.to_csv(file_name, index=False)
    print("Split Done!")
