"""
    Object used for storing stock data and creating variables
        - Not used for model building 
    Takes in actual stock data and predicted stock data
"""
import pandas as pd

class StockData:
    def __init__(self, actual_data, features, is_categrorical=False, predicted_data=None):
        self.actual_data = actual_data
        self.features = features
        self.predicted_data = predicted_data
        self.train_data = None
        self.test_data = None

    def get_data(self):
        return self.actual_data
    
    def train_test_split(self, test_size=0.2):
        self.train_data = self.actual_data[:int(len(self.actual_data) * (1 - test_size))]
        self.test_data = self.actual_data[int(len(self.actual_data) * (1 - test_size)):]
        return self.train_data, self.test_data

    # -- -- -- -- -- -- -- -- Create Columns for Testing -- -- -- -- -- -- -- -- # 
        # Create column that shows the change in stock price for a given time period.
        # Create column that shows if the stock price increased or decreased
        #  
        
        
    # -- -- -- -- -- -- -- -- Evaluate Predictive data -- -- -- -- -- -- -- -- # 
    # TODO: Profit function
        # Use the model's buy/sell predictions to calculate profit

    def hitRatio(actual, predicted):
        hits = 0
        for i in range(len(actual)):
            if actual[i] < 0 and predicted[i] < 0: # If both are negative
                hits += 1
            elif actual[i] > 0 and predicted[i] > 0: # If both are positive
                hits += 1
        return hits / len(actual)
    
    def evaluationMetrics():
        # Returns appropriate metrics for predicted values
        return -1
    # -- -- -- -- -- -- -- -- Data visualization -- -- -- -- -- -- -- -- #
    # def graph_data(self, )