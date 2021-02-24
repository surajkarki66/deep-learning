import tensorflow as tf

# Gradient Tapes
x = tf.ones((2, 2))
#print(x)

with tf.GradientTape() as t:
    t.watch(x)  # records
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
    

# Derivative of z with respect to the original input tensor x
dz_dx = t.gradient(z, x)
#dz_dy = t.gradient(z, y)
print(dz_dx)
#print(dz_dy)
#test
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0


# To Compute multiple gradients
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y
dz_dx = t.gradient(z, x)
dz_dy = t.gradient(z, y)

print(dz_dx)
print(dz_dy)


# Higher Order Differentiation(Gradients)
x = tf.Variable(1.0)

with tf.GradientTape() as t:
    with  tf.GradientTape() as t2:
        y = x * x * x
        # Compute the gradient inside the 't' context manager
  # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)

d2y_dx2 = t.gradient(dy_dx, x)

print(d2y_dx2)