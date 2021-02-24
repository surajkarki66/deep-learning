import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import model
from face_detection import get_detected_face

class FaceRecognition:
    def __init__(self):
        self.TRAINING_DATA_DIRECTORY = "./dataset/training"
        self.VALIDATION_DATA_DIRECTORY = "./dataset/testing"
        self.EPOCHS = 50
        self.BATCH_SIZE = 32
        self.NUMBER_OF_TRAINING_IMAGES = 320
        self.NUMBER_OF_TESTING_IMAGES = 196
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224
        self.model = model()
        self.training_generator = None
    
    @staticmethod
    def training_graph(history):
        plot_dir = "graph_plots"
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plt.savefig(os.path.join(plot_dir, "model_accuracy.png"))

    @staticmethod
    def data_generator():
        img_data_generator = ImageDataGenerator(
        rescale=1./255,
        # horizontal_flip=True,
        fill_mode="nearest",
        # zoom_range=0.3,
        # width_shift_range=0.3,
        # height_shift_range=0.3,
        rotation_range=30
    )
        return img_data_generator


    def train(self):
        self.training_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size = self.BATCH_SIZE,
            class_mode = 'categorical'
        )

        validation_generator = FaceRecognition.data_generator().flow_from_directory(
            self.VALIDATION_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            class_mode='categorical'
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer= tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-2 / self.EPOCHS),
            metrics=['accuracy']
        )
      

        history = self.model.fit(
            self.training_generator,
            steps_per_epoch=self.NUMBER_OF_TRAINING_IMAGES//self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=validation_generator,
            shuffle=True,
            validation_steps=self.NUMBER_OF_TESTING_IMAGES//self.BATCH_SIZE
        )

        FaceRecognition.training_graph(history)


    @staticmethod
    def load_saved_model(model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    
    def save(self, model_name):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.model.save(os.path.join(model_path, model_name))
        class_names = self.training_generator.class_indices
        class_names_file_reverse = model_name[:-3] + "_class_names_reverse.npy"
        class_names_file = model_name[:-3] + "_class_names.npy"
        np.save(os.path.join(model_path, class_names_file_reverse), class_names)
        class_names_reversed = np.load(os.path.join(model_path, class_names_file_reverse), allow_pickle=True).item()
        class_names = dict([(value, key) for key, value in class_names_reversed.items()])
        np.save(os.path.join(model_path, class_names_file), class_names)


    @staticmethod
    def model_predict(image_path, model_path, class_names_path):
        class_name = ""
        face_array, face = get_detected_face(image_path)
        model = tf.keras.models.load_model(model_path)
        face_array = face_array.astype('float32')
        input_face = np.expand_dims(face_array, axis=0)
        #result = model(input_face)
       # print(result)
        result = model.predict_classes(input_face)
        result = np.argmax(result, axis=1)
        index = result[0]

        classes = np.load(class_names_path, allow_pickle=True).item()
        print(classes, type(classes), classes.items())
        if type(classes) is dict:
            for k, v in classes.items():
                if k == index:
                    class_name = v

        return class_name

        
        
