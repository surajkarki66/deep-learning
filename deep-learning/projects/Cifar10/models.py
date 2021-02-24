import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19

def scratch_model(input_shape=None, num_classes=None):
	try:
		x_input = tf.keras.layers.Input(shape=input_shape)

		conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu',
										kernel_constraint=tf.keras.constraints.MaxNorm(3))(x_input)
		drop1 = tf.keras.layers.Dropout(0.2)(conv1)

		conv2 = tf.keras.layers.Conv2D(
			32, (3, 3), padding='same', activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(3))(drop1)

		maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
		flatten = tf.keras.layers.Flatten()(maxpool)
		dense1 = tf.keras.layers.Dense(
			512, activation='relu', kernel_constraint=tf.keras.constraints.MaxNorm(3))(flatten)
		drop2 = tf.keras.layers.Dropout(0.5)(dense1)
		dense2 = tf.keras.layers.Dense(
			num_classes, activation='softmax')(drop2)

		model = tf.keras.models.Model(inputs=x_input, outputs=dense2)

		return model

	except ValueError:
		print('Please provide to Input either a `shape` or a `tensor` argument. Note that `shape` does not include the batch dimension.')

def vgg19(input_shape=None, num_classes=None):
	try:
		model = VGG19(include_top=False, input_shape=input_shape)
		# mark loaded layers as non trainable
		for layer in model.layers:
			layer.trainable = False

		# new classifier layer
		flat = tf.keras.layers.Flatten()(model.layers[-1].output)
		dense = tf.keras.layers.Dense(
			128, activation='relu', kernel_initializer='he_uniform')(flat)
		output = tf.keras.layers.Dense(
			num_classes, activation='softmax')(dense)

		# define the model
		model = tf.keras.models.Model(inputs=model.inputs, outputs=output)

		return model

	except ValueError:
		print('Please provide to Input either a `shape` or a `tensor` argument. Note that `shape` does not include the batch dimension.')
