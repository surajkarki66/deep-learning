import os

from face_recognition import FaceRecognition
from face_detection import get_detected_face

if __name__ == "__main__":
    model_name = "face_recognition.h5"
    image_path = 'ronaldo.jpg'
    face_recognition = FaceRecognition()
    face_recognition.train()
    face_recognition.save(model_name)
   # model = FaceRecognition.load_saved_model(os.path.join("model", model_name))
   # f = FaceRecognition.model_predict(image_path, os.path.join("model", model_name),
                                         #os.path.join("model", "face_recognition_class_names.npy"))
    #print(f"This is {f}")
    
    
