from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


DCNN_model = load_model('DCNN_model.h5')

testImagePaths = [
    r"Data\test\crack\test_crack.jpg", r"Data\test\missing-head\test_missinghead.jpg", r"Data\test\paint-off\test_paintoff.jpg"]


classes = ['crack', 'missing-head', 'paint-off']


for img_path in testImagePaths:
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 


    # predict the class
    predictions = DCNN_model.predict(img_array)
    predicted_class = np.argmax(predictions)  # find the class with highest probability
    confidence = predictions[0][predicted_class]


    # display result
    print(f"Image: {img_path}")
    print(f"Predicted classification: {classes[predicted_class]} with confidence {confidence:.2f}")


    # display image with the prediction
    plt.imshow(img)
    plt.title(f"Predicted classification: {classes[predicted_class]} ({confidence:.2f})")
    plt.axis('off')
    plt.show() 