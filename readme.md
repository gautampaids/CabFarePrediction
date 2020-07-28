Cab Fare Prediction
----------------------------
Python

1. Open the notebook "Cab Fare Prediction.ipynb" from jupyter lab or from notebook
2. Set the directory of train_cab and test in the same dir where this notebook is loaded

ANN

1. Load the file ann.py in any python editor or run the file from the cmd "python ann.py"
2. The folders CLeansedData should be in the same directory

R

1. load the file Fare_Prediction.R in RStudio
2. Set the working directory to the directory where this file is located so that the program reads the data properly

There is a ProjectReport.pdf which gives more details on every step being followed in the process

Also the models are persisted in disk as
Cab_Fare_Prediction_RF.pkl -> python random forests model
final_linear_model_using_R.rds -> linear model using R
final_model_using_RF.rds -> Random Forests model using R
final_model_using_xgb.rds -> Xgboost model using R
cab_fare_ANN_model.h5 -> Winning ANN model in python
