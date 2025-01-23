import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
import matplotlib.pyplot as plt

class RNN:
    def __init__(self, input_shape, output_shape, seq_length=10):
        """
            input_shape: shape of data received will be tuple for [timesteps, features]
            output_shape: shape of the output the model will produce (i.e. price at given point, local min, local max, etc.)
            seq_length: sets the length of sequences used as input, model will look at 10 previous time steps to predict the next value
        """
        #TODO: Add parameter for neurons, output shape
        self.seq_length = seq_length
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(output_shape)
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    def save_model(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)
        print(f"Model loaded from {file_path}")

    @staticmethod
    def create_sequences(data, target, seq_length=10):
        X = []
        y = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(target[i+seq_length])
        return np.array(X), np.array(y)

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    def test(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        print(f'Mean Squared Error: {mse}')
        return predictions

if __name__ == "__main__":
    # Get Data and features
    file_path = 'data/AAPL/AAPL_data.csv'
    data = pd.read_csv(file_path)
    features = ['Close', 'Volume', 'Open', 'High', 'Low']
    target = 'Close'

    if not all(column in data.columns for column in features + [target]):
        print(f"Error: The CSV file must contain the following columns: {', '.join(features + [target])}")


    data.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[features])

    seq_length = 5
    X, y = RNN.create_sequences(scaled_features, data[target].values, seq_length=seq_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(features)))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(features)))

    rnn = RNN(input_shape=(seq_length, len(features)), output_shape=1)
    rnn.train(X_train, y_train, X_test, y_test, batch_size=32, epochs=10)

    predictions = rnn.test(X_test, y_test)