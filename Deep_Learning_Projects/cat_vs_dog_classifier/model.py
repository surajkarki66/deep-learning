import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg19 import VGG19


def simple_model(input_shape=None, classes = None):
	try:
		x_input = tf.keras.layers.Input(shape=input_shape)
		x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x_input)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = tf.keras.layers.Dropout(0.25)(x)

		x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = tf.keras.layers.Dropout(0.25)(x)

		x = tf.keras.layers.Conv2D(128,(3,3),activation='relu')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
		x = tf.keras.layers.Dropout(0.25)(x)

		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.Dropout(0.25)(x)
		x = tf.keras.layers.Dense(classes, activation='softmax')(x)
		model  = tf.keras.models.Model(inputs = x_input, outputs = x)

		return model

	except ValueError:
		print('Please provide to Input either a `shape` or a `tensor` argument. Note that `shape` does not include the batch dimension.')



# VGG19
def vgg19_model(input_shape=None, classes=None):
	try:
		model =  VGG19(include_top=False, input_shape=input_shape)
		# mark loaded layers as non trainable
		for layer in model.layers:
			layer.trainable = False

		# new classifier layer
		flat = Flatten()(model.layers[-1].output)
		dense = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat)
		output = Dense(classes, activation='softmax')(dense)

		#define the model
		model = Model(inputs=model.inputs, outputs=output)

		return model

	except ValueError:
			print('Please provide to Input either a `shape` or a `tensor` argument. Note that `shape` does not include the batch dimension.')

