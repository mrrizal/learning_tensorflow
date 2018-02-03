# codingan untuk bikin mirror
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# load image
dir_name = os.path.dirname(os.path.realpath(__file__))
file_name = dir_name + '/asset/MarshOrchid.jpg'
image = mpimg.imread(file_name)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    reverse = tf.reverse_v2(x, [1])
    reverse = session.run(reverse)
    result = tf.concat([x, reverse], 1)
    result = session.run(result)

plt.imshow(result)
plt.show()
