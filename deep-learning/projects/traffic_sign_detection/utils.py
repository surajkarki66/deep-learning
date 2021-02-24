import cv2
import numpy as np
import os

def load_image(path):
    count = 0
    images = []
    class_no = []
    my_list = os.listdir(path)
    print("Total Classes: ", len(my_list))
    noOfClasses=len(my_list)
    print("Importing Classes.....")
    
    for x in range(0, len(my_list)):
        myPicList = os.listdir(path+"/"+str(count))
        for y in myPicList:
            curImg = cv2.imread(path+"/"+str(count)+"/"+y)
            images.append(curImg)
            class_no.append(count)
        print(count, end =" ")
        count +=1
    print(" ")
    images = np.array(images)
    class_no = np.array(class_no)

    return images, class_no, noOfClasses