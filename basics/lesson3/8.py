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

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    height, witdh, channel = image.shape
    result1 = tf.slice(image, [0, 0, 0], [height, int(witdh/2), channel])
    reverse = tf.reverse_v2(result1, [1])
    result = tf.concat([result1, reverse], 1)
    result = session.run(result)


plt.imshow(result)
plt.show()