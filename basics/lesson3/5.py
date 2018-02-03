# versi sederhana dari 4.py
# pertama ubah image menjadi maatrix
# kedua, transpose matrix (gambar jadi miring ke kiri)
# ketika, reverse matrix dari kiri ke kanan (gambar jadi miring ke kanan/sesuai
# arah jarum jam)
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
    x = tf.reverse_v2(x, [1])
    image = session.run(x)

plt.imshow(image)
plt.show()

