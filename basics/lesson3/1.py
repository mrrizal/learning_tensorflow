import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
file_name = dir_path + '/asset/MarshOrchid.jpg'

# load image
image = mpimg.imread(file_name)

# print image shape
# hasilnya (5528, 3685, 3), artinya gambar memiliki ukuran 5528 x 3685 pixel dan 3 colors `deep`
print(image.shape)

# print image
plt.imshow(image)
plt.show()
