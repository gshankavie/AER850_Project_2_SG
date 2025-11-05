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
