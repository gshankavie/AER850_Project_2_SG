# %%
import sys
import numpy as np
#import pandas as pd
#print("hello")
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU


# %%



# Step 1: data processing
# define image shape and channel and batch size
image_width, image_height, image_channel, batch_size = 500, 500, 3, 16;

# %%
# define image data directories
training_data = r"Data\train"
validation_data = r"Data\valid"

# create training data image augmentation object
train_data_aug = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    #rotation_range = 40,
    #horizontal_flip = True
    )
