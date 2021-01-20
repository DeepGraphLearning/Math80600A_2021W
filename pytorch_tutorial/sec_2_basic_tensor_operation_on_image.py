import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def operation_compute_mean_along_an_axis(image):
    ########## Compute the mean value along an axis ##########
    channel_mean = image.float().mean(dim=2)
    print(channel_mean.shape)
    plt.imshow(channel_mean, cmap="gray")
    plt.show()
    return


def operation_slice_first_half_of_image(image):
    ########## Slice the first half of an axis ##########
    horizontal_crop = image[:, :256, :]
    print(horizontal_crop.shape)
    plt.imshow(horizontal_crop)
    plt.show()
    return


def operation_transpose(image):
    ########## Transpose (exchange) two axes ##########
    transposition = image.transpose(0, 1)
    print(transposition.shape)
    plt.imshow(transposition)
    plt.show()
    return


def operation_repeat(image):
    ########## Extend a tensor. Repeat the tensor to form a batch of images. ##########
    def plot(data, labels=None, num_sample=5):
        n = min(len(data), num_sample)
        for i in range(n):
            plt.subplot(1, n, i + 1)
            plt.imshow(data[i], cmap="gray")
            plt.xticks([])
            plt.yticks([])
            if labels is not None:
                plt.title(labels[i])

    batch = image.unsqueeze(0).repeat(3, 1, 1, 1)
    print(batch.shape)
    plot(batch)
    plt.show()
    return


def operation_reshape(image):
    ########## Reshape a tensor. The tensor is converted into a long thin matrix. ##########
    flat = image.flatten(0, 1)
    print(flat.shape)
    plt.imshow(flat)
    plt.show()
    return


if __name__ == '__main__':
    ########## Load image into numpy array ##########
    np_image = np.array(Image.open("lenna.png"))
    print('np_image shape:', np_image.shape)

    ########## Each element corresponds to a pixel in image ##########
    ########## Generally ranges 0 to 255 ##########
    ########## This image ranges 3 to 255 ##########
    print('np_image:', np_image)
    print('ranges from', np.min(np_image), 'to', np.max(np_image))

    ########## Transform from numpy array to pytorch tensor ##########
    image = torch.as_tensor(np_image)

    ########## Display the image ##########
    plt.imshow(image)
    plt.show()

    ########## Show some simple statistics about the data ##########
    print('image shape:', image.shape)
    print('image dim: ', image.ndim)
    print('image data type', image.dtype)

    ########## Play around with some simple operations on the image ##########
    operation_compute_mean_along_an_axis(image)
    operation_slice_first_half_of_image(image)
    operation_transpose(image)
    operation_repeat(image)
    operation_reshape(image)
