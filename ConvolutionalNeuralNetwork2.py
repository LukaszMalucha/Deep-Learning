### Convolutional Neural Network  - Cat or Dog ???





##################################################################### Building the CNN ################################################



### Importing Libraries

from keras.models import Sequential     # initialize
from keras.layers import Convolution2D  # Convolution layer
from keras.layers import MaxPooling2D   # Pooling
from keras.layers import Flatten        # Flattening
from keras.layers import Dense          # Fully connected layers for ANN

### Initializing the CNN

classifier = Sequential()

### Step 1 - Convolution 32x3x3

classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))        ## 32 - number of feature detectors; color image is 3d array; relu to not get any negative values


### Step 2 - Max Pooling

classifier.add(MaxPooling2D(pool_size = (2,2)))

### Step 2b - add additional convolutional layer for better result (from 50% to 80% accuracy)

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))             ## No input shape as it was already done
classifier.add(MaxPooling2D(pool_size = (2,2)))

### Step 3 - Flattening

classifier.add(Flatten())

### Step 4 - Full connection

# Hidden layer - 128 as a experience guess

classifier.add(Dense(activation = 'relu', units = 128))

# Output layer - one node

classifier.add(Dense(activation = 'sigmoid', units = 1))


### Compile the CNN Model

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



############################################################ Fitting the CNN to the images ###########################################

### From https://keras.io/preprocessing/image/   apply some random transformations on image dataset

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)



############################################################## Single Prediction with CNN ##########################################

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/FILE', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)                  ## add extra dimension as predict method expects 4                               
result = classifier.predict(test_image)
training_set.class_indices
if result [0][0]  == 1:
        prediction = 'dog'
else:
        prediction = 'cat'























