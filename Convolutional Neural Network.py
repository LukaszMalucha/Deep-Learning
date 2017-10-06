#### Convolutional Neural Network   -is it Cat or Dog?

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

######################################## Initiating Convolutional Neural Network ########################################################

# Importing the Keras libraries and packages
from keras.models import Sequential             ## initialize neural network
from keras.layers import Convolution2D          ## add convlutional layers
from keras.layers import MaxPooling2D           ## Pooling layers
from keras.layers import Flatten                ## Flattening into vector
from keras.layers import Dense                  ## Add ANN layers


### Initialize the CNN

classifier = Sequential()

### First layer of CNN

classifier.add(Convolution2D(32,                                    ## Feature Detector - number of filters (usually start with 32)
                              (3,                                    ## Feature Detector - rows
                              3),                                    ## Feature Detector - columns       
                             input_shape =(64,64,3),                ## Convert all images to the same size (dims,dims,channels) - order opposite in tensorflow tahn in theano
                              activation = 'relu'))                 ## activation function - rectifier function
                               
                               
### Pooling layer (to prevent overfitting)           
classifier.add(MaxPooling2D(pool_size = (2,2)))


### To improve accuracy over 80%  - add additional CNN

classifier.add(Convolution2D(32,(3,3),activation = 'relu'))   ## no need for input shape 
classifier.add(MaxPooling2D(pool_size = (2,2)))                             
                             
### Flattening into vector for Artificial Neural Network
classifier.add(Flatten())
                
                
########################################### Building Artificial Neural Network ########################################################
                                                       
### Fully connecting
                               
classifier.add(Dense(activation = 'relu', units = 128))     ## adding hidden layer dim as an average between input/out layers - activating relevant 'neurons' with rectifier function
classifier.add(Dense(activation = 'sigmoid', units = 1))    ## output layer  - sigmoid for binary, softmax for multi    


### Compiling the CNN
classifier.compile(optimizer = 'adam',                      ## stochastic grade
                   loss = 'binary_crossentropy',            ## crossentropy as a best choice (binary instead of categorical)
                   metrics = ['accuracy'])                  ## most common performance metric                   



#################################################### Fitting CNN to the images ########################################################

### https://keras.io/preprocessing/image/   - flow_from_directory method

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale = 1./255,                           ### left as it is on offcial url
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)                            ## resacale pixels

training_set = train_datagen.flow_from_directory('C:\\Users\\MaximusMinimus\\Desktop\\Machine Learning A-Z\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\training_set',       ## where do we extract images from
                                                 target_size = (64, 64),       ## dimensions
                                                 batch_size = 32,              ## weights will be updated after every 32 images
                                                 class_mode = 'binary')        ## two categories - cat & dog

test_set = test_datagen.flow_from_directory('C:\\Users\\MaximusMinimus\\Desktop\\Machine Learning A-Z\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\dataset\\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')



classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,                             ## 8000 images 
                         nb_epoch = 25,                                        ## 25 cycles
                         validation_data = test_set,
                         nb_val_samples = 2000)

### To improve accuracy  - add additional CNN












