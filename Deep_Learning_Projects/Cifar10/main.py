from nn import NN
from data import preprocessing, visualize


if __name__ == "__main__":
    #visualize()
    train_x, train_y, test_x, test_y, num_classes = preprocessing()
    input_shape = train_x[0].shape

    nn = NN(input_shape=input_shape, num_classes=num_classes)
    nn.summary()
    nn.compile(learning_rate=0.01, optimizer='sgd', loss='categorical_crossentropy')
    history = nn.fit(x = train_x, y = train_y, batch_size = 32, validation_data=(test_x, test_y))

    nn.save_model('scratch_model.h5')

    
   