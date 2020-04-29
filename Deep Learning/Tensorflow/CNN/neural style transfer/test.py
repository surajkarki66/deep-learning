import tensorflow as tf

model = tf.keras.applications.vgg19.VGG19(include_top=True)
print(model)