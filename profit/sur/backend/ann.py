import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from profit.sur import Surrogate

class ANNSurrogate(Surrogate):
    def __init__(self):
        self.trained = False
        pass
    
    def train(self, x, y):
        """Fits a artificial neural network to input points x and 
           model outputs y with scale sigma_f and noise sigma_n"""
        
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=1, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='linear'))

        self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        self.model.fit(x, y, epochs=128, batch_size=16)
        self.trained = True
    
    
    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma 
           and updates the inverted covariance matrix for the GP via the 
           Sherman-Morrison-Woodbury formula"""
        
        raise NotImplementedError()
        
    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')

        fpred = []
        fpred.append(self.model.predict(x))
        fpred.append(np.zeros_like(fpred[0]))  # no uncertainty information from ANN
        return fpred
        