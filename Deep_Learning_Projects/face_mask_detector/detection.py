import cv2
import os
import numpy as np
import time

from tensorflow.keras.models import load_model


class LiveDetection:
    """ Live Detection interface for driver sleep detection """
    def __init__(self):

        face_casc_path = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_casc_path)

        self.label = ['With_Mask', 'Without_mask']

        self.model = load_model('models/mask_detector_mobilenetv2_transfer_learning.h5')
        self.path = os.getcwd()
        self.cap = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL


    def start(self):
        """ Start the video frame """
        while(True):
            ret, frame = self.cap.read()
            height,width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 5,
                minSize = (30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

            for (x,y,w,h) in faces:
                face = frame[y:y+h,x:x+w]
                face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                face = cv2.resize(face,(224,224))
                face= face/255
                face =  face.reshape(224,224,-1)
                face = np.expand_dims(face,axis=0)
                predict = self.model.predict(face)
                confidence = np.amax(predict)
                prediction = np.argmax(predict)
                prediction = self.label[prediction]
                text = "Confidence: " + str(round(confidence * 100, 1 )) + "%"
                if prediction == self.label[0]:
                    cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 3 )
                    cv2.putText(frame, self.label[0], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),3,cv2.LINE_AA)
                    cv2.putText(frame, text, (x,y+h+1), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 0,0),2,cv2.LINE_AA)

                if prediction == self.label[1]:
                    cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,0,255) , 3 )
                    cv2.putText(frame, self.label[1], (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),3,cv2.LINE_AA)
                    cv2.putText(frame, text, (x,y+h+1), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 0,0),2,cv2.LINE_AA)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    l = LiveDetection()
    l.start()
