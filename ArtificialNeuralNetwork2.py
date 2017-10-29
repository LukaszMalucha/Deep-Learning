### Artificial Neural Network

### Installing Theano, Tensorflow, Keras


################################################################### Data Preprocessing ###################################################


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data - countries and gender

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]                               ## avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


########################################################### Creating Artificial Neural Network #######################################


### Import Keras

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout   ### Dropout regularization to reduce overfitting - randomized turning of neurons


### Initializing the ANN for classification problem

classifier = Sequential()


### Adding the INPUT layer and 1st HIDDEN Layer:
### Rectifier function for hidden l. and sigmoid function for output layer
### N of nodes in hidden layer = avg(input layer + output layer) 6 = (11 + 1)/2

classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_dim = 11))
classifier.add(Dropout(p = 0.1))


classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dropout(p = 0.1))
### Adding the output layer

classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

### Compiling the ANN - stochastic gradient descent:

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


### Fit the ANN to the Training set:
        
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)        


################################################################# Predictions of the Model ##########################################

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


### Single Prediction - use 2d array with double brackets (Scaling data with sc.transform)

new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)

## Customer doesn't leave the bank


################################################################### Evaluating the Model ############################################


### Importing Tools - wrapping sckit into Keras model

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

### Implementing function - ANN classifier

def build_classifier():
      classifier = Sequential()
      classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform', input_dim = 11))
      classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
      classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
      classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
      return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

## 10 accuracies of 10 tesfold in k-fold cross-validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


################################################################### Parameter Tuning ################################################


from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

### Implementing function - ANN classifier

def build_classifier():
      classifier = Sequential()
      classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform', input_dim = 11))
      classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'glorot_uniform'))
      classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))
      classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
      return classifier
classifier = KerasClassifier(build_fn = build_classifier)

## GridSearch

parameters = {'batch_size': [25, 32], 'nb_epoch': [100, 500], 'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

## Wait couple of hours............

## To get better result:  change kernel_initializer from 'uniform' to 'glorot_uniform'





























