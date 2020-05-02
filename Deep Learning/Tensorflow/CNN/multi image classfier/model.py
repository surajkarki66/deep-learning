import tensorflow as tf


def model(input_shape=(224, 224, 3), classes=3):
    x_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, kernel_size=3,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.),
                                activity_regularizer=tf.keras.regularizers.l2(0.),input_shape=input_shape)(x_input)

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.),
                                activity_regularizer=tf.keras.regularizers.l2(0.))(x)
    
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.),
                              activity_regularizer=tf.keras.regularizers.l2(0.))(x)

    x = tf.keras.layers.Dropout(0.05)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model  = tf.keras.models.Model(inputs = x_input, outputs = x)
    model.summary()
    
    return model

