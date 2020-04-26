import tensorflow as tf


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', input_shape=(64, 64, 3), activation='relu')
        self.maxpool1= tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.maxpool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.maxpool3 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(6)


    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x
