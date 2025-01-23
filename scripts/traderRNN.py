import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class StockTraderRNN:
    def __init__(self, name, window_size=60, epochs=50, batch_size=64):
        """
        Initializes an instance of the StockTraderRNN class with key parameters.

        :param name: (str) Name of the model instance.
        :param window_size: (int) Number of time steps to look back for each prediction window, default is 60.
        :param epochs: (int) Number of training iterations through the entire dataset, default is 50.
        :param batch_size: (int) Number of samples processed before updating the model, default is 64.
        """
        # Name of the model instance.
        self.name = name
        
        # Window size for the RNN model, representing the look-back period for predictions.
        self.window_size = window_size
        
        # Number of epochs or complete passes through the dataset for training.
        self.epochs = epochs
        
        # Batch size, determining the number of samples processed before updating weights.
        self.batch_size = batch_size
        
        # Scaler for normalizing input data to the (0,1) range to improve RNN training stability.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Placeholder for the neural network model, initialized later with specific layers and configurations.
        self.model = None
        
        # List to store features that are used as input to the model for training.
        self.features = []

    def preprocess_data(self, df, features=['Close', 'Volume']):
        """
        Preprocesses the financial data for the RNN.
        :param df: DataFrame containing financial data (e.g., OHLCV)
        :param features: List of column names to use as features for training
        :return: X and y arrays for training, and the fitted scaler
        """
        self.features = features
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size:i])
            future_price = df['Close'].values[i]
            current_price = df['Close'].values[i - 1]
            
            if future_price > current_price * 1.02:
                y.append(2)  # Buy
            elif future_price < current_price * 0.98:
                y.append(0)  # Sell
            else:
                y.append(1)  # Hold

        X, y = np.array(X), np.array(y)
        return X, y
    def create_model(self, input_shape,
                      lstm_units=50,
                      dense_units=3,
                      comp_loss='categorical_crossentropy', 
                      comp_optimizer='adam', 
                      comp_metrics=['accuracy']): 
        """
        Builds the RNN model. Is ran by train function
        :param input_shape: Shape of the input data (time_steps, features)

        # TODO Add descriptions of loss function, optimizer, and metrics


        :param comp_loss: Loss function for the model
        :param comp_optimizer: Optimizer for the model
        :param comp_metrics: Metrics to evaluate the model
        """

        """
        #TODO: Change comp_metrics to weighted 
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = {0: 2.0, 1: 1.0, 2: 2.0}  # Adjusted weights for classes
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, class_weight=class_weights)
        """
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=dense_units, activation='softmax'))

        model.compile(optimizer=comp_optimizer, loss=comp_loss, metrics=comp_metrics)
        self.model = model
    def train(self, df, X=None, y=None):
        """
        Trains model on preprocessed data or dataframe. Runs create_model
        :param df: DataFrame containing historical stock data
        :param X: Features for training
        :param y: Labels for training
        """
        if X is None or y is None:
            X, y = self.preprocess_data(df)
        X_train = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        y_train = tf.keras.utils.to_categorical(y, num_classes=3)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.create_model(input_shape)
        
        history = self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        return history
    def make_prediction(self, recent_data):
        """
        Makes a buy/sell/hold prediction based on recent data.
        :param recent_data: Most recent historical data (window_size length)
        :return: 'Buy', 'Sell', or 'Hold' decision
        """
        scaled_input = self.scaler.transform(recent_data)
        scaled_input = np.reshape(scaled_input, (1, scaled_input.shape[0], scaled_input.shape[1]))
        
        prediction = self.model.predict(scaled_input)
        decision = np.argmax(prediction)
        
        if decision == 0:
            return "Sell"
        elif decision == 1:
            return "Hold"
        else:
            return "Buy"
    def plot_price_predictions(self, df, X, y):
        """
        Plots actual vs. predicted price movements.
        :param df: DataFrame containing stock data
        :param X: Features for testing
        :param y: Labels for testing
        """
        X_test = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        predictions = np.argmax(self.model.predict(X_test), axis=1)
        
        plt.scatter(df['Timestamp'][self.window_size:], predictions, label='Predicted Actions')
        plt.scatter(df['Timestamp'][self.window_size:], y, label='Actual Actions', alpha=0.5)
        plt.xlabel('Timestamp')
        plt.ylabel('Action')
        plt.legend()
        plt.title('Buy/Sell/Hold Predictions vs. Actual')
        plt.show()
    def evaluate(self, X, y):
        """
        Evaluates the model's performance on test data.
        
        :param X: Features for testing, an array of shape (samples, time_steps, features)
        :param y: Labels for testing, an array of shape (samples,)
        :return: Model accuracy
        """
        # Reshape `X` to ensure it matches the required input shape for the model.
        # The input shape for LSTM models is (samples, time_steps, features).
        # `X.shape[0]` is the number of samples,
        # `X.shape[1]` is the number of time steps (window size),
        # `X.shape[2]` is the number of features.
        X_test = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
        
        # Convert the labels `y` to categorical format.
        # Since this model predicts three classes (Buy, Sell, Hold), the labels need to be one-hot encoded.
        # For example, if `y` has values [2, 0, 1], this step converts them to:
        # [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        # This encoding allows the model to evaluate each prediction across all categories.
        y_test = tf.keras.utils.to_categorical(y, num_classes=3)
        
        # Evaluate the model on the test data using the compiled loss and metrics. (Not recursive, using in-built evaluate function)
        # `self.model.evaluate()` will return the loss and any specified metrics (e.g., accuracy).
        # This allows us to see how well the model generalizes to new, unseen data. 
        loss, accuracy = self.model.evaluate(X_test, y_test)

        return accuracy
    def save_model(self, path):
        """
        Saves the trained model to a file.
        :param path: Path to save the model
        """
        # Save the model to the specified path
        self.model.save(path + self.name + '.h5')
        # TODO: Save the model params to a file
        # self.model
        # self.name
        # self.window_size
        # self.epochs
        # self.batch_size
        # self.scaler
        # self.model
        # self.features
        # self.features
