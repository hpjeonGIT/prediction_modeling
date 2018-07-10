import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ReduceLROnPlateau
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


# REF: https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#print(os.environ["CUDA_VISIBLE_DEVICES"])
# this doesn't affect the current running Python

CWD = os.getcwd()
series  = pd.read_csv(CWD+'/'+'some.csv',header=None)
#series  = pd.read_csv(CWD+'/'+'vsin.csv',header=None)
#series  = pd.read_csv(CWD+'/'+'sin_wave.csv',header=None)
#series = series[series.index%5 ==0]
#series.index= for i in range(len(series))
#aaa = [i for i,j in zip(series[0], range(len(series))) if j%5]
#series =pd.DataFrame(aaa)
# normalize
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(series.values)
series = pd.DataFrame(scaled)

window_size = 50

series_s = series.copy()
for i in range(window_size):
    series = pd.concat([series, series_s.shift(-(i+1))], axis = 1)
    


series.dropna(axis=0, inplace=True)
nrow = round(0.7*series.shape[0])

train = series.iloc[:nrow, :]
test = series.iloc[nrow:,:]

from sklearn.utils import shuffle
#train = shuffle(train)
train_X = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]
train_X = train_X.values
train_y = train_y.values
test_X = test_X.values
test_y = test_y.values


train_X = train_X.reshape(train_X.shape[0],train_X.shape[1],1)
test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(input_shape = (window_size,1), output_dim= window_size, return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.8))
model.add(Dense(1))
model.add(LeakyReLU())
model.add(Activation("linear"))
model.compile(loss="mse", optimizer="adam")
model.summary()

start = time.time()
model.fit(train_X,train_y,batch_size=256,nb_epoch=15,validation_split=0.1)
print("> Compilation Time : ", time.time() - start)


preds = model.predict(test_X)
preds = scaler.inverse_transform(preds)
actuals = scaler.inverse_transform(test_y[:,np.newaxis])

mean_squared_error(actuals,preds)
plt.plot(actuals)
plt.plot(preds)
plt.show()

def moving_test_window_preds(n_future_preds):
    preds_moving = []                                    # Use this to store the prediction made on each test window
    moving_test_window = [test_X[0,:].tolist()]          # Creating the first test window
    moving_test_window = np.array(moving_test_window)    # Making it an numpy array
    for i in range(n_future_preds):
        preds_one_step = model.predict(moving_test_window) # Note that this is already a scaled prediction so no need to rescale this
        preds_moving.append(preds_one_step[0,0]) # get the value from the numpy 2D array and append to predictions
        preds_one_step = preds_one_step.reshape(1,1,1) # Reshaping the prediction to 3D array for concatenation with moving test window
        moving_test_window = np.concatenate((moving_test_window[:,1:,:], preds_one_step), axis=1) # This is the new moving test window
    preds_moving = scaler.inverse_transform(np.array(preds_moving)[:,np.newaxis])
    return preds_moving


preds_moving = moving_test_window_preds(int(window_size*1.5))
plt.plot(actuals)
plt.plot(preds_moving)
plt.show()
