import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class App:
    def __init__(self):
        self.IMG_DIM = (64, 64)
        self.gesture_names = {
            0: 'Zero',
            1: 'One',
            2: 'Two',
            3: 'Three',
            4: 'Four',
            5: 'Five'
        }
        self.model = load_model('./model/ResNet-50_handsign.h5')

    def run(self):
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            ret, frame = cap.read()
            cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
            roi = frame[100:500, 100:500]
            img = cv2.resize(roi, self.IMG_DIM)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img.astype('float32')/255.0
            pred_array = self.model.predict(img)
            pred = np.argmax(pred_array)
            score = 'Accuracy: ' + \
                str(float("%0.2f" % (max(pred_array[0]) * 100)))
            color = (0, 0, 255)

            cv2.putText(
                frame, self.gesture_names[pred], (500, 50), font, 1.4, color, 2)
            cv2.putText(frame, score, (80, 90), font, 1.4, (255, 23, 2), 2)
            cv2.imshow('Hand Gesture', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    hand_gesture = App()
    hand_gesture.run()

