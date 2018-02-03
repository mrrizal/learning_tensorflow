# codingan untuk ngejungkir balikin gambar
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
    x = tf.reverse(x, [-3])
    image = session.run(x)

plt.imshow(image)
plt.show()

