import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = np.load('./datasets/crude_oil_scaled_data.npy', allow_pickle=True)

model = Sequential()
model.add(LSTM(50, input_shape=(30, 1), activation='tanh'))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.summary()

fit_hist = fit_hist = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), shuffle=False)
model.save('./models/crude_oil_modle_1.h5')
plt.plot(fit_hist.history['loss'], label='loss')
plt.plot(fit_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()