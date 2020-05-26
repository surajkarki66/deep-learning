import os
import tensorflow as tf
import PIL
import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from data import loading_data
from main import Model

def image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img =   cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
  	
    return img
   

if __name__ == "__main__":
    model = Model(input_shape=(28, 28, 1))
    train_ds, test_ds = loading_data()
    model.compile(learning_rate=0.01, optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(epochs=15, train_ds=train_ds, test_ds=test_ds)
    #filename='images/9.jpg'
    #img = image(filename)
    #pred = nn.predict(img)
    #final_pred = np.argmax(pred)
    #print(pred)
    #print(final_pred)
    model.save('digit_model.h5')







    
   







