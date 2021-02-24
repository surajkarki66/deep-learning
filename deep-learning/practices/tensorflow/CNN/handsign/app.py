import os
import tensorflow as tf
import PIL
import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from data import load_data
from main import NN

def image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    img =   cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    img = img.reshape(1, 64, 64, 3)
    img = img / 255.0
  	
    return img
   

if __name__ == "__main__":
    nn = NN()
    train_ds, test_ds = load_data()
    nn.train(epochs=100, train_ds=train_ds, test_ds=test_ds)
    filename='images/5.jpg'
    img = image(filename)   
    pred = nn.predict(img)
    final_pred = np.argmax(pred)
    print(pred)
    print(final_pred)
    nn.save(name='handsign1', save_format='tf')
    a = tf.keras.models.load_model('handsign')	
    #print(a.get_weights())
    pred = a.predict(img)
    print(pred)
    pred = np.argmax(pred)
    print(pred)
    







    
   







