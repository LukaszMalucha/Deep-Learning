### Mixed Deep Learning Model: Unsupervised + Supervised 


################################################ Part 1 - Identify the Frauds with the Self-Organizing Map ############################

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
frauds = np.concatenate((mappings[(8,7)], mappings[(3,1)]), axis = 0)        ## coords of white squares
frauds = sc.inverse_transform(frauds)                                        ## inversing normalization 



####################################################### Part 2 - Applying Supervised Deep Learning ###############################

### Creating the matrix of features that contains information regarding all the bank customers
customers = dataset.iloc[:, 1:].values   ## except the first one (cust id.)

### Creating the dependent variable
is_fraud = np.zeros(len(dataset))    ## initialize vector with zeros 
for i in range(len(dataset)):        ## change zero to one for each customer that cheated 
    if dataset.iloc[i,0] in frauds:  
        is_fraud[i] = 1

### Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

################################################################### Creating ANN ##############################################

### Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

### Initialising the ANN
classifier = Sequential()

### Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

### Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

### Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

### Predicting the probabilities of frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)      ##  Array - cust. id + probability; axisi 1 for horizontal concat.  
y_pred = y_pred[y_pred[:, 1].argsort()]                                       ##  sort probabilities (second column)





























