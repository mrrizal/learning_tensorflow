# tugas pertama
import numpy as np
import tensorflow as tf

data = np.random.randint(10, size=1000)
print(data)

x = tf.constant(data, name='x')
y = tf.Variable((5*(data*data)) - (3*data) + 15, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    print(sess.run(y))
