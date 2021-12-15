# from matplotlib import pyplot as plt
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
# first_image = mnist.test.images[0]
# first_image = np.array(first_image, dtype='float')
# pixels = first_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()



import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

with Image.open("hopper.jpg") as im:
    im.rotate(0)

img=Image.open("hopper.jpg")

img2 = Image.open('hopper.jpg').convert('L')



tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

first_image2=mnist.test.images[3]

first_image = np.array(first_image2, dtype='float')
pixels = first_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# # plt.imshow(first_image, cmap='gray')
# plt.show()

a = np.mgrid[:5, :5][0]

third_image=np.fft.fftshift(np.fft.fft2(pixels))
# for x in range(0,8):
#     for y in range(0,8):
#         third_image[10+x,10+y]=0

filterSize=10
for x in range(0, filterSize):
    for y in range(0, 28):
        third_image[ x, y] = 0
        third_image[27-x, y] = 0
for x in range(0, 28):
    for y in range(0, filterSize):
        third_image[x, y] = 0
        third_image[ x,27 - y] = 0

second_image=np.fft.fft2(third_image)
second_imageAbs=abs(second_image)


#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(1,2)

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(pixels, cmap='gray')
axarr[1].imshow(second_imageAbs, cmap='gray')


plt.show()



# # second_image=np.fft.fft2(im)
# # second_image=np.fft.fftshift(np.fft.fft2(img2))
# second_image=np.fft.fftshift(np.fft.fft2(pixels))
#
#
# second_imageAbs=np.log10(abs(second_image))
# # pixels = second_image.reshape((28, 28))
# plt.imshow(second_imageAbs, cmap='gray')
#
# plt.show()