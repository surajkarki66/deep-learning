import tensorflow as tf
import random
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



from utils import load_image


def spliting(test_size, validation_size):
    path = 'dataset'
    image_dimension = (32, 32, 3)
    images, class_no, noOfClasses = load_image(path)
    X_train, X_test, y_train, y_test = train_test_split(images, class_no, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

        
    y_train = to_categorical(y_train,noOfClasses)
    y_validation = to_categorical(y_validation,noOfClasses)
    y_test = to_categorical(y_test,noOfClasses)
    ############################### TO CHECK IF NUMBER OF IMAGES MATCHES TO NUMBER OF LABELS FOR EACH DATA SET
    print("Data Shapes")
    print("Train")
    print(X_train.shape,y_train.shape)
    print("Validation")
    print(X_validation.shape,y_validation.shape)
    print("Test")
    print(X_test.shape,y_test.shape)
    assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
    assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
    assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
    assert(X_train.shape[1:]==(image_dimension))," The dimesions of the Training images are wrong "
    assert(X_validation.shape[1:]==(image_dimension))," The dimesionas of the Validation images are wrong "
    assert(X_test.shape[1:]==(image_dimension))," The dimesionas of the Test images are wrong"

    return X_train, y_train, X_test, y_test, X_validation, y_validation, noOfClasses


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


def build_model(classes=43):
    image_dimension = (32, 32, 3)
    no_of_Filters=60
    size_of_Filter=(5,5) # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
                         # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2=(3,3)
    size_of_pool=(2,2)  # SCALE DOWN ALL FEATURE MAP TO GERNALIZE MORE, TO REDUCE OVERFITTING
    no_of_Nodes = 500   # NO. OF NODES IN HIDDEN LAYERS

    model= Sequential()
    model.add((Conv2D(no_of_Filters,size_of_Filter,input_shape=(image_dimension[0],image_dimension[1],1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # DOES NOT EFFECT THE DEPTH/NO OF FILTERS

    model.add((Conv2D(no_of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_of_Nodes,activation='relu'))
    model.add(Dropout(0.5)) # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(classes,activation='softmax')) # OUTPUT LAYER
    # COMPILE MODEL
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def model_evaluation(X_test, y_test, model):
    score = model.evaluate(X_test,y_test,verbose=0)
    print('Test Score:',score[0])
    print('Test Accuracy:',score[1])


def plot(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training','validation'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training','validation'])
    plt.title('Acurracy')
    plt.xlabel('epoch')
    plt.show()
   



if __name__ == "__main__":
    test_size = 0.2    # if 1000 images split will 200 for testing
    validation_size = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
    
    X_train, y_train, X_test, y_test, X_validation, y_validation, noOfClasses = spliting(test_size, validation_size)
    image_dimension = (32, 32, 3)
    X_train=np.array(list(map(preprocessing,X_train)))  # TO IRETATE AND PREPROCESS ALL IMAGES
    X_validation=np.array(list(map(preprocessing,X_validation)))
    X_test=np.array(list(map(preprocessing,X_test)))
    cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)]) # TO CHECK IF THE TRAINING IS DONE PROPERLY

    ############################### ADD A DEPTH OF 1 ############################
    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
    X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


    ############################### AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
    dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                                height_shift_range=0.1,
                                zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                                shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                                rotation_range=10)  # DEGREES
    dataGen.fit(X_train)
    batches= dataGen.flow(X_train,y_train,batch_size=20)  # REQUESTING DATA GENRATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
    X_batch,y_batch = next(batches)

    # TO SHOW AGMENTED IMAGE SAMPLES
    fig,axs=plt.subplots(1,15,figsize=(20,5))
    fig.tight_layout()

    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(image_dimension[0],image_dimension[1]))
        axs[i].axis('off')
    plt.show()


    # Training Generator
    training_generator = dataGen.flow(
        X_train,
        y_train,
        batch_size=50
    )

    model = build_model(classes=noOfClasses)
    model.summary()

    history = model.fit(
        training_generator,
        steps_per_epoch=2000,
        epochs=10,
        validation_data=(X_validation, y_validation),
        shuffle=True
    )
    plot(history)
    model.save('traffic_sign_detection.h5')

        
    	
    cv2.waitKey(0)


    
    
