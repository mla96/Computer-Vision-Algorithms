# Histogram manipulations and analyses for RGB images


from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np


# Create and display intensity histograms for each color channel (RGB)
def intensity_histograms(my_image):

    my_image_width, my_image_height = my_image.size
    color_values = [[], [], []]

    for i in range(my_image_width):
        for j in range(my_image_height):
            r, g, b = my_image.getpixel((i, j))
            color_values[0].append(r)
            color_values[1].append(g)
            color_values[2].append(b)
    color_values[0].sort()
    color_values[1].sort()
    color_values[2].sort()
    color_array = np.zeros((3, 256), dtype=int)

    for i in range(len(color_values)):
        for j in range(len(color_values[i])):
            color_array[i, color_values[i][j]] += 1

    for i in range(color_array.shape[0]):
        color_rgb = [0, 0, 0, 1]
        color_rgb[i] = 1
        plt.bar(range(color_array.shape[1]), color_array[i], color=tuple(color_rgb))
        plt.xlabel('Intensity (RGB Value)')
        plt.ylabel('Frequency (Pixels)')
        if color_rgb == [1, 0, 0, 1]:
            plt.title('Red Channel Histogram')
        elif color_rgb == [0, 1, 0, 1]:
            plt.title('Green Channel Histogram')
        elif color_rgb == [0, 0, 1, 1]:
            plt.title('Blue Channel Histogram')
        plt.show()


# Convert RGB image to greyscale image utilizing luminosity method
# I = 0.3 * R + 0.59 * G + 0.11 * B
def luminosity_method(my_image):

    my_image_width, my_image_height = my_image.size
    grey_values = []

    output_image = Image.new('L', (my_image_width, my_image_height))
    output_pixels = output_image.load()
    for i in range(my_image_width):
        for j in range(my_image_height):
            r, g, b = my_image.getpixel((i, j))
            grey_values.append(int(0.3*r + 0.59*g + 0.11*b))
            output_pixels[i, j] = int(0.3*r + 0.59*g + 0.11*b)

    grey_array = np.zeros(256, dtype=int)

    for i in range(len(grey_values)):
        grey_array[grey_values[i]] += 1

    plt.bar(range(len(grey_array)), grey_array, color='#000000')
    plt.xlabel('Intensity (Grayscale Value)')
    plt.ylabel('Frequency (Pixels)')
    plt.title('Grayscale Histogram - Luminosity Method')

    output_image.show()
    plt.show()


# Plot probability density function - histogram normalized by image size
def probability_cumulative_density_functions(my_image):  # For grayscale only

    my_image_width, my_image_height = my_image.size
    grey_values = []

    for i in range(my_image_width):
        for j in range(my_image_height):
            r, g, b = my_image.getpixel((i, j))
            grey_values.append(int(0.3*r + 0.59*g + 0.11*b))

    grey_array = np.zeros(256)

    for i in range(len(grey_values)):
        grey_array[grey_values[i]] += 1

    for i in range(len(grey_array)):
        grey_array[i] = grey_array[i]/(my_image_width * my_image_height)  # Divide by total number of pixels

    plt.plot(range(len(grey_array)), grey_array, color='#000000')  # Probability density function
    plt.xlabel('Intensity (Grayscale Value)')
    plt.ylabel('Normalized Frequency (Pixels/Total Pixels)')
    plt.title('Grayscale Histogram - Probability Density Function')
    plt.show()

    for i in range(1, len(grey_array)):
        grey_array[i] = grey_array[i - 1] + grey_array[i]

    plt.plot(range(len(grey_array)), grey_array, color='#000000')  # Cumulative density function
    plt.xlabel('Intensity (Grayscale Value)')
    plt.ylabel('Normalized Frequency (Pixels/Total Pixels)')
    plt.title('Grayscale Plot - Cumulative Density Function')
    plt.show()



def histogram_equalization(my_image):

    my_image_width_N, my_image_height_M = my_image.size
    G_grey_values = []

    for i in range(my_image_width_N):
        for j in range(my_image_height_M):
            r, g, b = my_image.getpixel((i, j))
            G_grey_values.append(int(0.3*r + 0.59*g + 0.11*b))

    # 1. For an N x M image of G gray-levels, initialize an array H of length G to 0.
    H_grey_array = np.zeros(256)

    # 2. Form the image histogram
    for i in range(len(G_grey_values)):
        H_grey_array[G_grey_values[i]] += 1

        # Then let g_min be the minimum g for which H[g] > 0
    g_min = [255, H_grey_array[0]]  # [intensity, number of pixels]
    for i in range(len(H_grey_array)):
        if i < g_min[0] and H_grey_array[i] > 0:
            g_min = [i, H_grey_array[i]]
    g_min = g_min[0]
    # print(g_min)

    # 3. Form the cumulative image histogram H_c:
    for i in range(1, len(H_grey_array)):
        H_grey_array[i] = H_grey_array[i - 1] + H_grey_array[i]
    # print(H_grey_array[0])

    # 4. Set T[g]
    T_grey_array = np.zeros(256)
    for i in range(len(H_grey_array)):
        T_grey_array[i] = round((((H_grey_array[i] - H_grey_array[g_min])/(my_image_width_N * my_image_height_M -
                          H_grey_array[g_min]))) * (len(H_grey_array) - 1))
    # print(T_grey_array)

    # 5. Rescan the image and write an output image with gray-levels g_q, setting g_q = T[g_p].
    output_image = Image.new('L', (my_image_width_N, my_image_height_M))
    output_pixels = output_image.load()
    output_pixel_list = []
    for i in range(my_image_width_N):
        for j in range(my_image_height_M):
            r, g, b = my_image.getpixel((i, j))
            G_grey_value = int(0.3*r + 0.59*g + 0.11*b)
            output_pixel_list.append(int(T_grey_array[G_grey_value]))
            output_pixels[i, j] = int(T_grey_array[G_grey_value])

    print(T_grey_array)

    equalized_histogram_array = np.zeros(256, dtype=int)

    for i in range(len(output_pixel_list)):
        equalized_histogram_array[output_pixel_list[i]] += 1

    plt.bar(range(len(equalized_histogram_array)), equalized_histogram_array, color='#000000')  # Equalized histogram
    plt.xlabel('Intensity (Grayscale Value)')
    plt.ylabel('Frequency (Pixels)')
    plt.title('Equalized Grayscale Histogram')

    output_image.show()
    plt.show()
