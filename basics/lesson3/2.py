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
    # jika tinda ingin dibalik
    # x = tf.transpose(x, perm=[0, 1, 2])
    x = tf.transpose(x, perm=[1, 0, 2])
    session.run(model)
    result = session.run(x)

# gambar sebelum di rubah
print("Gambar sebelum dirubah {}".format(image.shape))
print("Gambar setelah dirubah {}".format(result.shape))
plt.imshow(result)
plt.show()
