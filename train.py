# anaconda modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras import optimizers



import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Global variables
models = [
  LinearRegression(), # метод наименьших квадратов
  RandomForestRegressor(n_estimators=100, max_features ='sqrt'), # случайный лес
  KNeighborsRegressor(n_neighbors=6), # метод ближайших соседей
  SVR(kernel='linear'), # метод опорных векторов с линейным ядром
  # LogisticRegression() # логистическая регрессия
  ]

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }


def start (mod_df: pd.DataFrame, show: bool = True):

  trg = mod_df[['sigma']]
  trn = mod_df.drop (['sigma'], axis=1)

  x_train, x_test, y_train, y_test = train_test_split (trn, trg, test_size=0.4,
                                               random_state=7)

  sc_x = StandardScaler()
  sc_x.fit(x_train)
  x_train_sc = sc_x.transform(x_train)
  x_test_sc = sc_x.transform(x_test)

  sc_y = StandardScaler()
  sc_y.fit(y_train)
  y_train_sc = sc_y.transform(y_train)
  y_test_sc = sc_y.transform(y_test)

  # To be used later while visualizing results
  actual_creep = np.transpose(y_test)

  # Building the Neural Network

  model = Sequential()
  model.add(Dense(units = 15, kernel_initializer = 'normal', activation = 'tanh', input_dim = 16))
  model.add(Dense(units = 30, kernel_initializer = 'normal', activation = 'tanh'))
  model.add(Dense(units = 45, kernel_initializer = 'normal', activation = 'tanh'))
  model.add(Dense(units = 40, kernel_initializer = 'normal', activation = 'tanh'))
  model.add(Dense(units = 30, kernel_initializer = 'normal', activation = 'tanh'))
  model.add(Dense(units = 20, kernel_initializer = 'normal', activation = 'tanh'))
  model.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'tanh'))
  model.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'tanh'))

  sgd = optimizers.SGD(lr=0.03, decay=1e-7, momentum=0.9, nesterov=True)
  model.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = ['mean_squared_error'])

  # Training the model and predicting the results
  history = model.fit(x_train_sc, y_train_sc, batch_size = 256, shuffle=True, epochs = 2000)
  y_nn_pred_sc = model.predict(x_test_sc)

  # Determining the model's accuracy
  r2_nn = r2_score(y_test_sc, y_nn_pred_sc)
  mse_nn = mean_squared_error(y_test_sc, y_nn_pred_sc)
  mae_nn = mean_absolute_error(y_test_sc, y_nn_pred_sc)
  print('R\u00b2_score = ' + str(round(r2_nn, 2)))
  print('mean_squared_error = ' + str(round(mse_nn, 2)))
  print('mean_absolute_error = ' + str(round(mae_nn, 2)))

  # Scaling up the outputs back to original
  y_nn_pred = sc_y.inverse_transform(y_nn_pred_sc)

  # Visualizing the accuracy of predicted values
  nn_predicted_creep = np.transpose(y_nn_pred)

  # Plotting graphs for 0.2% Proof Strength and Tensile Strength
  fig, (ax0) = plt.subplots(1,figsize=(16,7))

  ax0.scatter(nn_predicted_creep, actual_creep, color = 'hotpink', s=18)
  x3 = np.linspace(0, 800, 1000)
  y3 = x3
  ax0.plot(x3, y3)
  ax0.set_title('Creep of alloy', fontsize = 20)
  ax0.set_xlabel('predicted_creep', fontsize = 14)
  ax0.set_ylabel('actual_creep', fontsize = 14)

  if (show):
    plt.show()