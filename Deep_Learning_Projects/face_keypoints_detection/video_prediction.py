####  TEST YOUR VIDEO FILE WITH THE MODEL  #####

import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time


model = load_model('simple_model.h5')  # <-- Saved model path


# input video file path
input_file = 'test.mp4'


# output file path
output_filename = 'testVideo_out.avi'


def get_points_main(img):

    def detect_points(face_img):
        me = np.array(face_img)/255
        x_test = np.expand_dims(me, axis=0)
        x_test = np.expand_dims(x_test, axis=3)
        y_test = model.predict(x_test)
        label_points = (np.squeeze(y_test)*48)+48

        return label_points

    # load haarcascade
    cascPath = os.path.dirname(cv2.__file__) + \
        "/data/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascPath)

    dimensions = (96, 96)

    try:
        default_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(default_img, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)

    except:
        return []

    faces_img = np.copy(gray_img)

    plt.rcParams["axes.grid"] = False

    all_x_cords = []
    all_y_cords = []

    for i, (x, y, w, h) in enumerate(faces):

        h += 10
        w += 10
        x -= 5
        y -= 5

        try:
            just_face = cv2.resize(gray_img[y:y+h, x:x+w], dimensions)
        except:
            return []
        cv2.rectangle(faces_img, (x, y), (x+w, y+h), (255, 0, 0), 1)

        scale_val_x = w/96
        scale_val_y = h/96

        label_point = detect_points(just_face)

        all_x_cords.append((label_point[::2]*scale_val_x)+x)
        all_y_cords.append((label_point[1::2]*scale_val_y)+y)

    final_points_list = []
    try:
        for ii in range(len(all_x_cords)):
            for a_x, a_y in zip(all_x_cords[ii], all_y_cords[ii]):
                final_points_list.append([a_x, a_y])
    except:
        return final_points_list

    return final_points_list


cap = cv2.VideoCapture(input_file)
ret, frame = cap.read()
height, width, channel = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (width, height))


frame_no = 0
while cap.isOpened():

    a = time.time()

    frame_no += 1
    ret, frame = cap.read()
    if frame_no > 75*30:
        break
    if frame_no in range(60*30, 75*30):
        points = get_points_main(frame)

        try:
            overlay = frame.copy()
        except Exception as e:
            print(e)
            break

        for point in points:

            cv2.circle(frame, tuple(point), 5, (255, 0, 0), -2)

        if len(points) != 0:
            o_line_points = [[12, 13], [13, 11], [11, 14], [14, 12], [12, 10], [11, 10], [10, 3], [12, 5], [11, 3], [
                10, 5], [10, 4], [10, 2], [5, 1], [1, 4], [2, 0], [0, 3], [5, 9], [9, 8], [8, 4], [2, 6], [6, 7], [7, 3]]
            num_face = len(points)//15

            for i in range(num_face):
                line_points = np.array(o_line_points) + (15*(i))

                the_color = (255, 0, 0)

                for ii in line_points:
                    cv2.line(overlay, tuple(points[ii[0]]), tuple(
                        points[ii[1]]), the_color, thickness=1)

        opacity = 0.5
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        out.write(frame)
        cv2.imshow('frame', frame)
        b = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
