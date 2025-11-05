from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


DCNN_model = load_model('DCNN_model.h5')

testImagePaths = [
    r"Data\test\crack\test_crack.jpg", r"Data\test\missing-head\test_missinghead.jpg", r"Data\test\paint-off\test_paintoff.jpg"]


classes = ['crack', 'missing-head', 'paint-off']

