# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
X_train = pd.read_csv('CleansedData/Xtrain.csv')
X_test = pd.read_csv('CleansedData/Xtest.csv')
y_train = pd.read_csv('CleansedData/ytrain.csv')
y_test = pd.read_csv('CleansedData/ytest.csv')

scoring = pd.read_csv("test/test_preprocessed.csv", parse_dates=True)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
scoring_sc = sc.transform(scoring)
y_train = sc.fit_transform(y_train)
y_test = sc.transform(y_test)

# Now let's make the ANN!
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from keras import metrics

# Initialising the ANN
model = tf.keras.Sequential([
                       tf.keras.layers.Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9),
#                       tf.keras.layers.Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'),
                       tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear')
                       ])

#tf.keras.layers.Dropout(rate=0.3)
#early_stop_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Compiling the ANN
model.compile(optimizer = 'rmsprop', loss = 'huber_loss', metrics = [metrics.mean_squared_error, metrics.mean_absolute_percentage_error])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 20, epochs = 50)

# Predicting the Test set results
y_pred_train = model.predict(X_train)

y_train_actual = sc.inverse_transform(y_train)
y_train_pred = sc.inverse_transform(y_pred_train)

def print_modelling_metrics(y_true,y_pred):
    mae = np.mean(np.abs(y_true-y_pred))
    print("Mean Absolute Error", mae)
    mape = np.mean(np.abs(y_true-y_pred/y_true)) * 100
    print("Mean Absolute Percentage Error", mape)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    print("Root Mean Squared", rmse)
    r2score = r2_score(y_train, y_pred_train)
    print("R2Score", r2score)
    evs = explained_variance_score(y_train, y_pred_train)
    print("EVS", evs)

print_modelling_metrics(y_train_actual, y_train_pred)

df = pd.DataFrame(np.column_stack([y_train_actual,y_train_pred]), columns = ["Actual","Predicted"])

y_pred_test = model.predict(X_test)

y_test_actual = sc.inverse_transform(y_test)
y_test_pred = sc.inverse_transform(y_pred_test)

print_modelling_metrics(y_test_actual, y_test_pred)
df_test = pd.DataFrame(np.column_stack([y_test_actual,y_test_pred]), columns = ["Actual","Predicted"])

#Residual Analysis
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred-y_train_actual, c='blue', marker = 'o', label = 'Training data')
plt.scatter(y_test_pred, y_test_pred-y_test_actual, c='orange', marker = '*', label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-3, xmax = 33, lw=2, color='k')
plt.xlim([-5,35])
plt.ylim([-25,15])
plt.show()

#Submission of the baseline model results to scoring set
scoring_fare_amount = model.predict(scoring_sc)
scoring_fare_amount = sc.inverse_transform(scoring_fare_amount)

scoring_fare_amount = pd.DataFrame(scoring_fare_amount, columns = ["predicted_fare_amount"])

pd.concat([scoring_fare_amount,scoring], axis = 1).to_csv("test/test_DL.csv")

#Saving the model
model.save('cab_fare_ANN_model.h5')
#model =  tf.keras.models.load_model('cab_fare_ANN_model.h5')

