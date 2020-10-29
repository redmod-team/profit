# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:10:48 2020
@author: khall
"""

import numpy as np 
import pandas as pd 
from pandas_datareader import DataReader 
from datetime import datetime 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 

spy = DataReader('SPY',  'yahoo', datetime(2013,1,1), datetime(2015,1,1)) #print(spy) spy_returns = pd.DataFrame(np.diff(np.log(spy['Adj Close'].values)))
# creates a table of dataframe countaining the values of SP&500 from (2013,1,1) to (2015,1,1)
# the tables' columns are: ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
# the first column is 'Date' though. It's however considered as an index and not a column.

# Extraction of the Close values
y_all = spy['Close'].values
    
# Plot of the Close values
plt.figure(figsize=(15,5))
plt.plot(y_all)
plt.ylabel('Close values')
plt.title('S&P500 Close Values')
plt.show()
plt.figure()
plt.hist(y_all, bins=30)


import GPy

x = np.array(range(503))+1
a = np.zeros(504)
a2 = np.zeros(504)
a3 = np.zeros(504)
y_all = spy['Close'].values
b = np.linspace(0,2,504)

for i in x:
    nm = i
    nm2 = i+1
    x = b[0:i].reshape((nm,1))
    y = y_all[0:nm].reshape((nm,1))
    x_new = b[0:nm2].reshape((nm2,1))
    
    kern = GPy.kern.Matern32(1,lengthscale= 0.01)
    m = GPy.models.GPRegression(x,y,kern,noise_var=0.001**2)
    m.optimize()
    e,f = m.predict(x_new)
    
    kern2 = GPy.kern.Matern52(1,lengthscale= 0.01)
    m2 = GPy.models.GPRegression(x,y,kern2,noise_var=0.001**2)
    m2.optimize()
    e2,f = m2.predict(x_new)
    
    kern3 = GPy.kern.Brownian(1)
    m3 = GPy.models.GPRegression(x,y,kern3,noise_var=0.001**2)
    m3.optimize()
    e3,f = m3.predict(x_new)
    
    a[i]=y_all[nm]-e[nm]
    a2[i]=y_all[nm]-e2[nm]
    a3[i]=y_all[nm]-e3[nm]

plt.figure()
plt.plot(b[5:i],a[5:i])
plt.title('Matern32')

plt.figure()
plt.plot(b[5:i],a2[5:i])
plt.title('Matern52')

plt.figure()
plt.plot(b[26:i],a3[26:i])
plt.title('Brownian Motion')

plt.figure()
plt.hist(a[5:i])
plt.title('Matern32')

plt.figure()
plt.hist(a2[5:i])
plt.title('Matern52')

plt.figure()
plt.hist(a3[26:i])
plt.title('Brownian Motion')
