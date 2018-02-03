import tensorflow as tf
import numpy as np

data = np.random.randint(10, size=1000)

x = tf.constant(data, name='x')
y = tf.Variable((5*(data*data)) - (3*data) + 15, name='y')

model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    result = sess.run(y)
    for index, i in enumerate(result):
        print('{}: {}'.format(index, i))
