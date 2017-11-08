### Recurrent Neural Network

################################################################# Data Preprocessing ############################################

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')            ## as a dataframe
training_set = dataset_train.iloc[:, 1:2].values                       ## transform to numpy array for keras (1:2 - upper bound is excluded but it'll create an array)  

### Feature Scaling with normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))                              ## all the prices will be between 0 and 1
training_set_scaled = sc.fit_transform(training_set)                   ## created separately from original training set


############################################# Creating a data structure with 60 timesteps and 1 output ###########################

### Network will observe 60 stock prices before time t and based on trends will try to predict next outcome

X_train = []                                                           ## input (60 days before)
y_train = []                                                           ## output (next day)
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)                ## transform into arrays for keras

### Reshaping - adding dimension for additional indicator (3d)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                ##   target    batch size -rows, -columns, number of indicators


################################################## Building Recurrent Neural Network #############################################


### Importing the Keras libraries and packages
from keras.models import Sequential   # for ANN
from keras.layers import Dense        # for output layer  
from keras.layers import LSTM         # for LSTM
from keras.layers import Dropout      # for regularisation


### Initialize RNN
regressor = Sequential()       


### Adding first LSTM layer and Dropout regularistation to remove overfitting risk

regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1], 1)))   ## number of units, return sequences = true forstacked LSTM, input shape of X_train from line 32
regressor.add(Dropout(0.2))              ## add dropout of 20% neurons that are getting ignored during each iteration


### Adding second LSTM layer + dropout

regressor.add(LSTM(units = 50,return_sequences = True))  ## no need to specify input for second layer
regressor.add(Dropout(0.2)) 

### Adding third LSTM layer + dropout

regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

### Adding fourth LSTM layer + dropout

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


### Adding the output layer

regressor.add(Dense(units = 1))   ## price prediction


### Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')           ## mean squared error as it's a regression, not classification


## Fitting the RNN to the Training set

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)               ## updating weights every 23 prices 




############################################## Part 3 - Making the predictions and visualising the results #####################


### Real stock price of 2017

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')           
real_stock_price = dataset_test.iloc[:, 1:2].values

### Predicted stock price for 2017

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)   ## concatinating original trainand test , concat. along vertical axis
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values          ## 60 previous stock prices as numpy array
inputs = inputs.reshape(-1,1)                                                        ## shape as numpy array
inputs = sc.transform(inputs)                                                        ## scaling

### 3D structure expected by neural network
X_test = []
for i in range(60, 80):                                              ## upper bound 80 as set has 20 financial days
    X_test.append(inputs[i-60:i, 0])                                 ## append with previous stock prices
X_test = np.array(X_test)                                           
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))   ## 3d with extra dim
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)  ## inverse scaling as regressor was train on scaled values


### Visualising Results

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()






























