import numpy as np
import cv2
import matplotlib.pyplot as plt


def plotImagePair(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        if name == "image":
            image[:,:, 0] = image[:,:, 0] * 255.0/image[:,:, 0].max()
            image[:,:, 1] = image[:,:, 1] * 255.0/image[:,:, 1].max()
            image[:,:, 2] = image[:,:, 2] * 255.0/image[:,:, 2].max()
            #image[:,:, 3] = image[:,:, 3] * 255.0/image[:,:, 3].max()
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if name == 'image':
            plt.imshow(image[:,:,0:3])
        plt.imshow(image)
    plt.show()