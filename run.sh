# Run in terminal:  sh run.sh

export TRAINING_DATA=input/train_10folds.csv
export TEST_DATA=input/test.csv
export FOLDS=10    # same as TRAINING_DATA
export MODEL=7    # select actual model

clear             # for clearing termninal window


# -------------------------------
# Step 1: searching for best baseline models ------
# -------------------------------
#python3 -m src.best_models
#python3 -m src.automl
#python3 -m src.tpot_pipeline

# -------------------------------
# Step 2: split in k folds ------
# -------------------------------
#python3 -m src.create_folds 10

# -------------------------------
# Step 3: Create new features ------
# -------------------------------
#python3 -m src.feature_selector

# -------------------------------
# Step 4: Select best hyperparameters ------
# -------------------------------
#python3 -m src.hyperparam_opt

# -------------------------------
# Step 5: Train with best hyperparameters ------
# -------------------------------
#python3 -m src.train

# -------------------------------
# Step 6: Predict ------
# -------------------------------
#python3 -m src.predict

# -------------------------------
# Step 6: Blend models ------
# -------------------------------
#python3 -m src.blender



#python3 -m src.utils
#python3 -m src.dispatcher




