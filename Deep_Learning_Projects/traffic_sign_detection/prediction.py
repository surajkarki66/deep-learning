import numpy as np
import pickle
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array


def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(path):
    img = cv2.imread(path)
    img = grayscale(img)
    img = equalize(img)
    img = cv2.resize(img, (32, 32))
    img = img/255
    while True:
        cv2.imshow("Processed Image", img)      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    img = img.reshape(1, 32, 32, 1)
    return img

def get_classname(class_no):
    data = pd.read_csv('labels.csv')
    classname= None 
    for class_id, class_name in data.values:
        if class_no == class_id:
            classname = class_name

    return classname

def predict(img):
    pickle_in=open("traffic_model.p","rb")  ## rb = READ BYTE
    model=pickle.load(pickle_in)

    prediction = model.predict(img)
    class_index = model.predict_classes(img)
    return prediction, class_index


if __name__ == "__main__":     
    filename = './images/straightOrLeft.jpg'
    img = preprocessing(filename)
    prediction, class_index = predict(img)
    probability_value =np.amax(prediction)
    print(probability_value)
    print(class_index)
    print(get_classname(class_index))

