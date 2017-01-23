import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import sys

# read test images

file_list = os.listdir('test_images')

for image_file in file_list:
    # reading in an image
    image = mpimg.imread(os.path.join('test_images', image_file))
    # printing out some stats and plotting
    print('This image is:', type(image), 'with dimensions:', image.shape)
    fig = plt.figure()
    fig.suptitle(image_file)
    plt.imshow(image)

plt.show()
