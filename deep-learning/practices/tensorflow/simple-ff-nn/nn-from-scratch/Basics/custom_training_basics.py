import tensorflow as tf
import matplotlib.pyplot as plt

# Variables
#Tensors in TensorFlow are immutable stateless objects

# Using Python state
x = tf.zeros([10, 10])
x += 2  # This is equivalent to x = x + 2, which does not mutate the original
        # value of x
print(x)

# Use tf.Variable for model weights
v = tf.Variable(1.0)

print(v.numpy())
v.assign(2.0)
print(v)


# Lets Fit a linear Model

# Define Model
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    
    def __call__(self,x):
        y_hat = self.W * x + self.b
        return y_hat


model = Model()
#print(model(9.0))

# Define a loss function ( mean squared error)
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

# Obtain training data

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

# Visualize
plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: %1.6f' % loss(model(inputs), outputs).numpy())

# Defining a training loop
# Backprop
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(outputs, model(inputs))
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)# w = w - learning_rate * dw
    model.b.assign_sub(learning_rate * db)# b = b - learning_rate * dw



model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(1000)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(outputs, model(inputs))

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()
