from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from xgboost import XGBRegressor


import numpy as np
import math
import pandas as pd



def nn_train_run(x_train, y_train, x_test, y_test):
    reg_val = 0.01
    model = tf.keras.models.Sequential([
  
  tf.keras.layers.Dense(250, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(250, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dropout(.2,),
  tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(5, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(5, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(5, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
    tf.keras.layers.Dropout(.2,),
  tf.keras.layers.Dense(150, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  tf.keras.layers.Dense(5, activation='relu', kernel_regularizer=keras.regularizers.l2(l=reg_val)),
  
  tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

    model.fit(x_train, y_train, validation_split=0.1, epochs=150)
    p_train = model.predict(x_test)
    return model, p_train

train_data = pd.read_csv('train.csv')
y_train = train_data.SalePrice
train_data = train_data.drop(['SalePrice'], axis=1)
train_data_encoded = pd.get_dummies(train_data.select_dtypes("object"))
train_data = train_data.drop(train_data.select_dtypes("object"), axis=1)

train_data = train_data.join(train_data_encoded)

train_data.fillna(0, inplace=True)
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)


(x_train, x_test, y_train, y_test) = train_test_split(train_data, y_train, test_size = .25)

model, p_train = nn_train_run(x_train, y_train, x_test, y_test)

print(model.evaluate(x_test, y_test))
mse = metrics.mean_squared_error(np.log(p_train), np.log(y_test))
rmse = math.sqrt(mse)
print(f"rmse {rmse}")




# test_data = pd.read_csv('test.csv')

# test_data_encoded = pd.get_dummies(test_data.select_dtypes("object"))
# test_data = test_data.drop(test_data.select_dtypes("object"), axis=1)

# test_data = test_data.join(test_data_encoded)

# test_data.fillna(0, inplace=True)
# scaler = MinMaxScaler()
# test_data = scaler.fit_transform(test_data)

# predictions = model.predict(test_data)

# final_pd = pd.DataFrame(test_data.Id, columns=['Id', 'SalePrice'])
# final_pd.SalePrice = predictions
# final_pd.to_csv('predictions_nn.csv', index=False)