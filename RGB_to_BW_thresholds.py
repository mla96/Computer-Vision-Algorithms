# Thresholding RGB images to BW


from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np


def binary_image_threshold(my_image, threshold):  # Threshold should be integer from 0 to 255

    if threshold < 0 or threshold > 255:
        return "Threshold value should be an integer between 0 and 255."

    my_image_width, my_image_height = my_image.size

    output_image = Image.new('L', (my_image_width, my_image_height))
    output_pixels = output_image.load()

    for i in range(my_image_width):
        for j in range(my_image_height):
            intensity = my_image.getpixel((i, j))
            if intensity >= threshold:  # Greater or equal to threshold
                output_pixels[i, j] = 255  # White
            else:
                output_pixels[i, j] = 0  # Black

    output_image.show()


def otsus_method(my_image):

    my_image_width, my_image_height = my_image.size
    grey_values = []

    output_image = Image.new('L', (my_image_width, my_image_height))
    output_pixels = output_image.load()

    for i in range(my_image_width):
        for j in range(my_image_height):
            grey_values.append(my_image.getpixel((i, j)))

    grey_array = np.zeros(256)

    for i in range(len(grey_values)):
        grey_array[grey_values[i]] += 1
    total_pixels = sum(grey_array)

    dot_product = np.dot([range(256)], grey_array)
    variance_list = np.zeros(256)
    max_interclass_variance = 0
    optimal_threshold = -1

    black_pixels = 0
    white_pixels = total_pixels
    cumulative_product = 0

    for threshold in range(0, 256):  # Threshold
        black_pixels += grey_array[threshold]
        white_pixels -= grey_array[threshold]

        if black_pixels == 0 or white_pixels == 0:
            continue

        cumulative_product += threshold * grey_array[threshold]  # Cumulative product = total black intensity
        white_intensity_pixel = (dot_product - cumulative_product) / white_pixels
            # (Total - total black intensity)/white pixels = total white intensity/white pixels =
            # white intensity per white pixel
        interclass_variance = black_pixels * white_pixels * ((cumulative_product / black_pixels) - white_intensity_pixel)**2
        variance_list[threshold] = interclass_variance
        if interclass_variance >= max_interclass_variance:
            optimal_threshold = threshold
            max_interclass_variance = interclass_variance

    intensity_array = []
    histogram_array = np.zeros(256, dtype=int)

    for i in range(my_image_width):  # Use optimal threshold to find pixel intensities relative to threshold
        for j in range(my_image_height):
            intensity = my_image.getpixel((i, j))
            if intensity >= optimal_threshold:  # Greater or equal to threshold
                intensity_array.append(255)  # White
                output_pixels[i, j] = 255
            else:
                intensity_array.append(0)  # Black
                output_pixels[i, j] = 0

    for i in range(len(intensity_array)):
        histogram_array[intensity_array[i]] += 1

    plt.bar(range(len(histogram_array)), histogram_array, color='#000000')  # Histogram for Otsu's Method
    plt.xlabel('Intensity (Grayscale Value)')
    plt.ylabel('Frequency (Pixels)')
    plt.title('Grayscale Histogram - Otsu`s Method')
    plt.show()

    plt.plot(range(len(variance_list)), variance_list, color='#000000')  # Inter-class Variance Plot
    plt.xlabel('Intensity (Grayscale Value)')
    plt.ylabel('Inter-class Variance (Intensity^2)')
    plt.title('Grayscale Inter-class Variance Plot')
    plt.show()

    print(max_interclass_variance)  # State inter-class variance
    print(optimal_threshold)  # Print intensity threshold

    output_image.show()  # Binary image for Otsu's Method
