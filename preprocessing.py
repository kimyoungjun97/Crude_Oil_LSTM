import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('./datasets/Clear_Crude_Oil_Data')
print(raw_data.info())

minmaxscaler = MinMaxScaler()
data = raw_data[:-30][['Price']]
scaled_data = minmaxscaler.fit_transform(data)
sequence_X = []
sequence_Y = []

for i in range(len(scaled_data)-30):
    _x = scaled_data[i:i+30]
    _y = scaled_data[i+30]
    sequence_X.append(_x)
    sequence_Y.append(_y)
    if i is 0:
        print(_x, '->', _y)

sequence_X = np.array(sequence_X)
sequence_Y = np.array(sequence_Y)

X_train, X_test, Y_train, Y_test = train_test_split(sequence_X, sequence_Y, test_size=0.2)
scaled_data = X_train, X_test, Y_train, Y_test

np.save('./datasets/crude_oil_scaled_data.npy', scaled_data)



