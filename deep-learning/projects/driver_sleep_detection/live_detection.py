import cv2
import os
import numpy as np
import time

from keras.models import load_model
from sounds.play import play
#os.environ['DISPLAY'] = ':0'

class LiveDetection:
    """ Live Detection interface for driver sleep detection """
    def __init__(self, driver_name):
        self.driver_name = driver_name

        face_casc_path = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(face_casc_path)

        l_eye_casc_path = os.path.dirname(cv2.__file__)+"/data/haarcascade_lefteye_2splits.xml"
        self.leye = cv2.CascadeClassifier(l_eye_casc_path)

        r_eye_casc_path = os.path.dirname(cv2.__file__)+"/data/haarcascade_righteye_2splits.xml"
        self.reye = cv2.CascadeClassifier(r_eye_casc_path)

        self.label = ['Close', 'Open']
                
       
        self.model = load_model('models/driver_sleep_detection.h5')
        self.path = os.getcwd()
        self.cap = cv2.VideoCapture(0)
        self.font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        self.count=0
        self.score=0
        self.thicc=2
        self.rpred=[99]
        self.lpred=[99]

        self.sounds_path = 'sounds/sounds_whoop.wav'

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
        
            left_eye = self.leye.detectMultiScale(gray)
            right_eye =  self.reye.detectMultiScale(gray)

            cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
                cv2.putText(frame, self.driver_name, (x,y), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
            

            for (x,y,w,h) in right_eye:
                r_eye=frame[y:y+h,x:x+w]
                self.count=self.count+1
                r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
                r_eye = cv2.resize(r_eye,(24,24))
                r_eye= r_eye/255
                r_eye=  r_eye.reshape(24,24,-1)
                r_eye = np.expand_dims(r_eye,axis=0)
                self.rpred = self.model.predict_classes(r_eye)
                if(self.rpred[0]==1):
                    self.label='Open' 
                if(self.rpred[0]==0):
                    self.label='Closed'
                break

            for (x,y,w,h) in left_eye:
                l_eye=frame[y:y+h,x:x+w]
                self.count=self.count+1
                l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
                l_eye = cv2.resize(l_eye,(24,24))
                l_eye= l_eye/255
                l_eye=l_eye.reshape(24,24,-1)
                l_eye = np.expand_dims(l_eye,axis=0)
                self.lpred = self.model.predict_classes(l_eye)
                if(self.lpred[0]==1):
                    self.label='Open'   
                if(self.lpred[0]==0):
                    self.label='Closed'
                break

            if(self.rpred[0]==0 and self.lpred[0]==0):
                self.score=self.score+1
                cv2.putText(frame,"Closed",(10,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
            else:
                self.score=self.score-1
                cv2.putText(frame,"Open",(10,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)
            
                
            if(self.score<0):
                self.score=0   
            cv2.putText(frame,'Score:'+str(self.score),(100,height-20), self.font, 1,(255,255,255),1,cv2.LINE_AA)

            if(self.score>10):
                # person is feeling sleepy so we beep the alarm
                cv2.imwrite(os.path.join(self.path,'image.jpg'),frame)
                try:	
                    play(self.sounds_path)
                except:
                    pass
                if(self.thicc<16):
                    self.thicc= self.thicc+2
                else:
                    self.thicc=self.thicc-2
                    if(self.thicc<2):
                        self.thicc=2
                cv2.rectangle(frame,(0,0),(width,height),(0,0,255),self.thicc) 
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()   
        cv2.destroyAllWindows()
