import tensorflow as tf


def build_model(input_shape=None, classes=None):
	try:
		input_data = tf.keras.layers.Input(shape=input_shape)

		x = tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), kernel_initializer='he_uniform')(input_data)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

		x = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform')(x)
		x = tf.keras.layers.Conv2D(64, (3, 3), kernel_initializer='he_uniform')(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform')(x)
		x = tf.keras.layers.Dense(10)(x)

		model = tf.keras.models.Model(inputs=input_data, outputs=x)

		model.summary()

		return model

	except ValueError:
		print('Please provide to Input either a `shape` or a `tensor` argument. Note that `shape` does not include the batch dimension.')

