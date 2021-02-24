import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from skimage import io

from model import model, define_model

class CatVsDog:
    def __init__(self):
        self.TRAINING_DATA_DIRECTORY = "./dataset/train"
        self.VALIDATION_DATA_DIRECTORY = "./dataset/validation"
        self.EPOCHS = 10
        self.BATCH_SIZE = 64
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224
        self.CHANNELS = 3
        self.NUMBER_OF_TRAINING_IMAGES = 3198
        self.NUMBER_OF_VALIDATION_IMAGES = 100

        self.model = define_model()
        self.training_generator = None
        self.validation_generator = None
  
    def data_preparation(self):
        train_data_generator = ImageDataGenerator(
            rotation_range=15,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        validation_data_generator = ImageDataGenerator(rescale=1./255)

        self.training_generator = train_data_generator.flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='binary'
        )

      
        self.validation_generator = validation_data_generator.flow_from_directory(
            self.VALIDATION_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='binary'
        )

        print("Classes")
        print(self.training_generator.class_indices)
        return self.training_generator, self.validation_generator


    def data_visualization(self):
        training_generator, _ = self.data_preparation()
        image_batch, label_batch = next(iter(training_generator))

        print(len(image_batch))
        for i in range(len(image_batch) - 1, len(image_batch)):
            image = image_batch[i]
            print(label_batch[i])
            plt.imshow(image)
            plt.show()

    def train(self):
        training_generator, validation_generator = self.data_preparation()
        self.model.summary()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(
            training_generator,
            steps_per_epoch=self.NUMBER_OF_TRAINING_IMAGES//self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=validation_generator,
            shuffle=True,
            validation_steps=self.NUMBER_OF_VALIDATION_IMAGES//self.BATCH_SIZE

        )

        self.model.save('catVsdog.h5')

        return history



