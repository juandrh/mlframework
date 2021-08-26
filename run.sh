export TRAINING_DATA=input/train_10folds.csv
export TEST_DATA=input/test.csv
export FOLD=10
export MODEL=1

clear
#python3 -m src.create_folds 10

python3 -m src.train
#FOLD=1 python3 -m src.train
#FOLD=2 python3 -m src.train
#FOLD=3 python3 -m src.train
#FOLD=4 python3 -m src.train



#python3 -m src.predict

# 5folds  0.7083991545362129
# 10folds 0.7081404817771173