# Artificial Neural Network

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding the categorical variables as factors (set factors as numeric)
dataset$Geography = as.numeric(factor(dataset$Geography,
                                      levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                   levels = c('Female', 'Male'),
                                   labels = c(1, 2)))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])

# Fitting ANN to the Training set - best package is h2o
install.packages('h2o')
library(h2o)
h2o.init(nthreads = -1)     ## initialize connection instance (-1 will optimize number of cores used)
model = h2o.deeplearning(y = 'Exited',                            ## dependent variable        
                         training_frame = as.h2o(training_set),   ## as h2o frame object
                         activation = 'Rectifier',                ## activation function
                         hidden = c(6,6),                         ## vector of number of hidden layers and nodes 5x5
                         epochs = 100,                            ## number of cycles
                         train_samples_per_iteration = -2)        ## -2 is default auto-tuning

# Predicting the Test set results
y_pred = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = (y_pred > 0.5)
y_pred = as.vector(y_pred)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
## Accuracy is 86%
