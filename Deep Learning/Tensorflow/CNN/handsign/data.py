import tensorflow as tf
import matplotlib.pyplot as plt
import h5py
import numpy as np

from model import Model



def preprocessing():
    train_dataset = h5py.File('data/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('data/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes


    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_data():
    train_x_orig, train_y, test_x_orig, test_y, classes = preprocessing()
    
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_orig/255.
    test_x = test_x_orig/255.
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_x[i], cmap=plt.cm.binary)
        plt.xlabel(train_y[i])
    plt.show()

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)

    return train_ds, test_ds


    