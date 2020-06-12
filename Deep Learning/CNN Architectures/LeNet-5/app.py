import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from main import Main

def load_data():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    return train_x, train_y, test_x, test_y


def visualize():
    train_x, _, _, _ = load_data()

    n = 6
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(330+1+i)
        plt.imshow(train_x[i])
    plt.show()


def preprocessing():
    train_x, train_y, test_x, test_y = load_data()
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')

    train_x = train_x/255.0
    test_x = test_x/255.0

    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    num_classes = test_y.shape[1]


    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(32)

    return train_ds, test_ds, num_classes

if __name__ == "__main__":
    m = Main()
    train_ds, test_ds, _ = preprocessing()
    m.train(train_ds = train_ds, test_ds = test_ds, epochs=1)
    m.save(name='LeNet-5')
    model = tf.keras.models.load_model('LeNet-5')
    model.summary()

