export TRAINING_DATA=input/train_10folds.csv
export TEST_DATA=input/test.csv
export FOLD=10    # same as TRAINING_DATA
export MODEL=4    # select actual model
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
python3 -m src.train

# -------------------------------
# Step 5: Predict ------
# -------------------------------
#python3 -m src.predict




#python3 -m src.dispatcher




