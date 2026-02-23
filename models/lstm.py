import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(series):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))

    X, y = [], []

    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    return model
