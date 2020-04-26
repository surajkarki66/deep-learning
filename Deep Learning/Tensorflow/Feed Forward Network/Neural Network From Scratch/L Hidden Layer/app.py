import  h5py
import numpy as np
import tensorflow as tf
from model import Model


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def preprocessing():
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    num_px = train_x_orig.shape[1]
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    return train_x, test_x, train_y, test_y, classes, num_px


if __name__ == "__main__":
    layer_dims = [12288, 10, 5, 1]
    train_x, test_x, train_y, test_y, classes, num_px = preprocessing()
    train_x = tf.cast(train_x, tf.float32)
    train_y = tf.cast(train_y, tf.float32)
    model = Model(layer_dims)
    parameters = model.fit(train_x, train_y, num_iterations = 2000, print_cost = True)
    
    pred_train = model.predict(train_x, train_y, parameters)
    pred_test = model.predict(test_x, test_y, parameters)

