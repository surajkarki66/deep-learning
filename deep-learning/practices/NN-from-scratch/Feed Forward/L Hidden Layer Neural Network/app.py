import h5py
import  numpy as np
from PIL import Image
import  matplotlib.pyplot as plt


from main import Main


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
    layers_dims = [12288, 20, 10, 1] #  4-layer model
    m = Main(layers_dims)
    
    train_x, test_x, train_y, test_y, classes, num_px = preprocessing()
    parameters = m.L_layer_model(train_x, train_y, layers_dims, num_epochs = 5000, print_cost = True, lambd=0, learning_rate=0.001)
    pred_train = m.predict(train_x, train_y, parameters)
    pred_test = m.predict(test_x, test_y, parameters)

    my_image = "cat.jpg" 
    my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

    fname = "images/" + my_image
    image = np.array(plt.imread(fname))
    my_image = ((np.array(Image.fromarray(image).resize((num_px,num_px), Image.ANTIALIAS)).reshape((1, num_px*num_px*3)))).T

    my_predicted_image = m.predict(my_image, my_label_y, parameters)

    plt.imshow(image)
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


        
