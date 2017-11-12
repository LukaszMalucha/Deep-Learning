### Self Organizing Map - Fraud Detector

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Importing dataset http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values                   # 0 - bank client application was approved, 1 - was rejected

### Feature Scaling for X (client features)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))        # range 0-1 for normalization
X = sc.fit_transform(X)


# Training Self Organizing Map

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)  # x,y for grid dimms ; input_len for x attributes; sigma default; l_r default
som.random_weights_init(X)                                                       # initialize weights with random small number
som.train_random(data = X, num_iteration = 100)


# Visualizing the results (higher mid, higher chance for outlayer - fraud)
from pylab import bone, pcolor, colorbar, plot, show
bone()                                ## window for a map
pcolor(som.distance_map().T)          ## matrix of all the node distances (T for transpose)
colorbar()                            ## color legend 
markers = ['o', 's']                  ## circles & squares
colors = ['r', 'g']                   ## colors
for i, x in enumerate(X):             ## i for row, x for customer attr. vector 
    w = som.winner(x)                 ## winning node
    plot(w[0] + 0.5,                  ## plot the marker into center of the winning node (0.5 to move to the middle)
         w[1] + 0.5,
         markers[y[i]],               
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',    ## no inside marker color for better visibilty 
         markersize = 10,
         markeredgewidth = 2)
show()


### Identify Frauds

mappings = som.win_map(X)                                                    ## all the mappings for all of the winning nodes (with associated customers)
frauds = np.concatenate((mappings[(1,4)], mappings[(8,3)]), axis = 0)        ## coords of white squares
frauds = sc.inverse_transform(frauds)                                        ## inversing normalization 
















