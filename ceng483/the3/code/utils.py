from skimage import io, color
from skimage.transform import rescale
import numpy as np
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def read_image(filename):
    img = io.imread(filename)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img


def cvt2Lab(image):
    Lab = color.rgb2lab(image)
    return Lab[:, :, 0], Lab[:, :, 1:]  # L, ab


def cvt2rgb(image):
    return color.lab2rgb(image)


def upsample(image):
    return rescale(image, 4, mode='constant', order=3)

def tensorshow(tensor,cmap=None):
    img = transforms.functional.to_pil_image(tensor/2+0.5)
    if cmap is not None:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)

def visualize_batch(inputs,preds,targets,save_path=''):
    inputs = inputs.cpu()
    preds = preds.cpu()
    targets = targets.cpu()
    plt.clf()
    bs = inputs.shape[0]
    for j in range(bs):
        plt.subplot(3,bs,j+1)
        assert(inputs[j].shape[0]==1)
        tensorshow(inputs[j],cmap='gray')
        plt.subplot(3,bs,bs+j+1)
        tensorshow(preds[j])
        plt.subplot(3,bs,2*bs+j+1)
        tensorshow(targets[j])
    if save_path != '':
        plt.savefig(save_path)
    else:
        plt.show(block=True)
