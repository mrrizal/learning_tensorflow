import tensorflow as tf

# di python biasa
x = 35
y = x + 5
print(y)

# di tensorf flow
x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')
print(y)

# untuk print value dari y, harus dijalankan oleh tensorflow session
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y))
