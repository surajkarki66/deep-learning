import numpy as np
import cv2
import tensorflow as tf
import pandas as pd


frameWidth= 640         
frameHeight = 480
brightness = 180
threshold = 0.75    
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model = tf.keras.models.load_model('digit_model.h5')

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def get_classname(value):
    data = pd.read_csv('labels.csv')
    classname= None 
    for class_id, class_name in data.values:
        if value == class_id:
            classname = class_name

    return classname

while True:
    # READ IMAGE
    success, imgOrignal = cap.read()

    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (28, 28))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 28,28, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    prediction = model.predict(img)
    prediction = tf.nn.softmax(prediction)
    probabilityValue =np.amax(prediction)
    class_id = np.argmax(prediction)
    class_name = get_classname(class_id)
    if probabilityValue > threshold:
        cv2.putText(imgOrignal,str(class_id)+" "+str(class_name), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
