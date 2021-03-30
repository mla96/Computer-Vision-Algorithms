# Functions for image smoothing filter application


from PIL import Image, ImageFilter
import copy
import numpy as np


# Apply 2D square filters of k^2 dimensions for some k_selection array of ks
def box_filter(my_image, k_selection):

    my_image_width, my_image_height = my_image.size
    output_image = Image.new('L', (my_image_width, my_image_height))
    output_pixels = output_image.load()

    intensity_image = np.zeros((my_image_width, my_image_height))  # matrix all intensities
    for i in range(my_image_width):
        for j in range(my_image_height):
            intensity_image[i, j] = my_image.getpixel((i, j))

    for k in k_selection:

        box_kernel = np.ones((k, k)) * (1/(k*k))
        k_range = int((k - 1) / 2)

        for i in range(k_range, my_image_height - k_range):  # image height within boundaries
            for j in range(k_range, my_image_width - k_range):  # image width within boundaries
                neighbors = intensity_image[i - k_range:i + k_range + 1,j - k_range:j + k_range + 1]
                output_pixels[i, j] = int(np.sum(box_kernel * neighbors))

        output_image.show()


# Apply 1D Gaussian filter
def gaussian_filter_1D(my_image):

    my_image_width, my_image_height = my_image.size
    output_image = Image.new('L', (my_image_width, my_image_height))
    output_pixels = output_image.load()

    intensity_image = np.zeros((my_image_width, my_image_height))  # matrix all intensities
    for i in range(my_image_width):
        for j in range(my_image_height):
            intensity_image[i, j] = my_image.getpixel((i, j))

    gaussian_kernel = np.array([0.03, 0.07, 0.12, 0.18, 0.20, 0.18, 0.12, 0.07, 0.03])
    gaussian_range = int((len(gaussian_kernel) - 1)/2)

    axes = [0, 1]
    for axis in axes:
        if axis == 1:
            intensity_image = np.transpose(intensity_image)
        intensity_image_source = copy.deepcopy(intensity_image)
        for i in range(gaussian_range, np.size(intensity_image, 0) - gaussian_range):  # image axis length
            for j in range(np.size(intensity_image, 1)):  # image axis length within boundaries
                neighbors = []
                for l in range(i - gaussian_range, i + gaussian_range + 1):  # axis length including neighbors
                    neighbors.append(intensity_image_source[l, j])
                intensity_image[i, j] = int(np.sum(gaussian_kernel * neighbors))

    intensity_image = np.transpose(intensity_image)
    for i in range(my_image_height):
        for j in range(my_image_width):
            output_pixels[i, j] = int(intensity_image[i, j])
    output_image.show()
