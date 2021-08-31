# Run in terminal:  sh run.sh

export TRAINING_DATA=input/train_10folds.csv
export TEST_DATA=input/test.csv
export FOLDS=10    # same as TRAINING_DATA
export MODEL=7    # select actual model
clear

# python3 -m src.best_models

# -------------------------------
# Step 1: split in k folds ------
# -------------------------------
#python3 -m src.create_folds 10

# -------------------------------
# Step 2: Create new features ------
# -------------------------------
#python3 -m src.feature_selector

# -------------------------------
# Step 3: Select best hyperparameters ------
# -------------------------------
#python3 -m src.hyperparam_opt

# -------------------------------
# Step 4: Train with best hyperparameters ------
# -------------------------------
#python3 -m src.train

# -------------------------------
# Step 5: Predict ------
# -------------------------------
#python3 -m src.predict

#python3 -m src.blender


#python3 -m src.dispatcher

#python3 -m src.utils

python3 -m src.automl
#python3 -m src.tpot_pipeline

#  0.71427606010154
# with type reduction  0.7091925651810564 310.4 s. 0.7085267794064445

#GBRegressor(input_matrix, learning_rate=0.1, max_depth=4, min_child_weight=13, n_estimators=100, n_jobs=1, objective=reg:squarederror, subsample=0.8, verbosity=0)
