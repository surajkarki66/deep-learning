import os
import tensorflow as tf
import glob
import numpy as np
import matplotlib.pyplot as plt

from utils import preproccess



def image(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(64, 64))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x)
    print('Input image shape:', x.shape)
  	
    return x

if __name__ == "__main__":
    model = tf.keras.models.load_model('handsign.h5')
   # model.summary()

    x_train, y_train, x_test, y_test, classes = preproccess()
    # evaluate
    print(x_test.shape)
    print(model.evaluate(x_test, y_test, batch_size=32))
    

   
