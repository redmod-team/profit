"""
Created: Mon Jul  8 11:47:38 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

from abc import ABCMeta, abstractmethod

class Surrogate(metaclass=ABCMeta):
    def __init__(self):
        pass
    
    @abstractmethod
    def train(self, x, y, sigma=None):
        """Trains the surrogate on input points x and model outputs y 
           with std. deviation sigma"""
        pass

    @abstractmethod
    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma 
           and updates the surrogate"""
        pass

    @abstractmethod
    def predict(self, x):
        """Predicts model output y for input x based on surrogate"""
        pass
