import numpy as np
# import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size=500
batch_xs, batch_ys = mnist.train.next_batch(batch_size)

batch1=batch_xs
# batch1[1]=batch_xs[2]

# print(batch1.shape[0])
# for x in range(0,batch1.shape[0]):
#     batch1[x]=abs(np.fft.fftshift(np.fft.fft2(batch_xs[x].reshape((28, 28))))).reshape(784)
#
#
# pixels = batch1[3].reshape((28, 28))

f, axarr = plt.subplots(2,2)
# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0][0].imshow(batch_xs[1].reshape((28, 28)), cmap='gray')
axarr[1][0].imshow(abs(batch_xs[1]).reshape((28, 28)), cmap='gray')
axarr[0][1].imshow(batch1[2].reshape((28, 28)), cmap='gray')
axarr[1][1].imshow(batch1[2].reshape((28, 28)), cmap='gray')
plt.show()

# second_image=np.fft.fftshift(np.fft.fft2(pixels))
#
#
# second_imageAbs=abs(second_image)
# # pixels = second_image.reshape((28, 28))
# plt.imshow(second_imageAbs, cmap='gray')
# plt.show()