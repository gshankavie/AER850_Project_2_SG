import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Step 1
image_width, image_height, image_channel, batch_size = 500, 500, 3, 32;

# image data
training_data = r"Data\train"
validation_data = r"Data\valid"

# image augmentation
train_data_aug = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    rotation_range = 20.0,
    horizontal_flip = True
    )
    
valid_data_aug = ImageDataGenerator(
    rescale = 1./255
    )

# generators
train_gen = train_data_aug.flow_from_directory(
    training_data,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'  
    )

valid_gen = valid_data_aug.flow_from_directory(
    validation_data,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
    )


# Step 2
DCNNmodel = Sequential()
DCNNmodel.add(Conv2D(16, (3, 3), strides=(1, 1), activation='relu',input_shape=(image_width, image_height, image_channel)))
DCNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
DCNNmodel.add(Conv2D(32, (3, 3), strides=(1, 1), activation='relu'))
DCNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
DCNNmodel.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
DCNNmodel.add(MaxPooling2D(pool_size=(2, 2)))

DCNNmodel.add(Flatten()),
DCNNmodel.add(Dense(128, activation = 'relu')), 
DCNNmodel.add(Dense(32, activation = 'relu'))
DCNNmodel.add(Dropout(0.1))
DCNNmodel.add(Dense(3, activation = 'softmax')) 

print(DCNNmodel.summary())


# Step 3
learning_rate = 1e-4
DCNNmodel.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
