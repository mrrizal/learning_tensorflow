"""
pada script kedua ini, kita akan memutar gambar 90 derajat
dengan cara menukar sumbu x dan y.

padah script sebelumnya, kita telah mengetahui bahwa gambar MarshOrchid
memiliki ukuran (5528, 3685, 3) = (x, y, z)

agar gambar berubah menjadi 90 derajat, yang perlu kita lakukan
adalah menukar nilai x dengan nilai y (menggunakan fungsi transpose matrix)
jadi natni matrix nya akan berupa
(y, x, z) = (3685, 5528, 3)
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# load image
dir_name = os.path.dirname(os.path.realpath(__file__))
file_name = dir_name + '/asset/MarshOrchid.jpg'
image = mpimg.imread(file_name)

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
    # putar gambar jadi 90 derajat ke kiri
    session.run(model)
    x = tf.transpose(x, perm=[1, 0, 2])
    image = session.run(x)

    # reverse pixel dari kiri ke kanan dan sebaliknya
    height, width, depth = image.shape
    x = tf.reverse_sequence(x, [width] * height, 1, batch_dim=0)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()
