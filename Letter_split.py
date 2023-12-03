import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os

def load_image(image_path):
    """
    Load an image, rescale it  and convert it to grayscale.
    Add noise.
    """
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (256, 64))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) # blur to be resistant to noise
    return gray_image


def detect_contour(gray_image):
    '''
    Otsu method and binary thresholding to make the constrat sharper 
    '''
    _, otsu_thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh
    
# def detect_contour(gray_image, lower_threshold = 100, upper_threshold = 255):
#     canny_edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
#     _, edges = cv2.threshold(canny_edges, 0, 255, cv2.THRESH_BINARY)
#     return edges

def plot_detect_contour(original_image, contour_image):
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(contour_image, cmap='gray')
    plt.title('Contour Image')
    plt.axis('off')

    plt.show()

def count_white_pixels_per_column(image):
    """
    Count the number of white pixels in each column of the image.( this histogram will be used to split letters)
    """
    height, width = image.shape
    white_pixel_counts = []
    for col in range(width):
        column = image[:, col]
        white_pixels = np.sum(column < 20)
        white_pixel_counts.append(white_pixels)

    return white_pixel_counts

def plot_histogram(white_pixel_counts):
    plt.bar(range(len(white_pixel_counts)), white_pixel_counts)
    plt.xlabel('Column Index')
    plt.ylabel('White Pixel Count')
    plt.title('White Pixel Counts Per Column')
    plt.show()

def slice_image_based_on_histogram(gray_image, white_pixel_counts, threshold=5):
    """
    Slice the image into smaller images based on the histogram of white pixel counts.
    """
    start_idx = None
    slices = []
    for idx, count in enumerate(white_pixel_counts):
        if count >= threshold and start_idx is None:
            start_idx = idx
        elif (count <= threshold or idx == len(white_pixel_counts) - 1) and start_idx is not None:
            slice_img = gray_image[:, start_idx:idx]
            slices.append(slice_img)
            start_idx = None
    return slices







# def find_limits(labels):
#     """
#     Find the column limits for each group of letters.
#     """
#     limits = {}
#     for label in np.unique(labels):
#         if label == 0:
#             continue  # Skip the background
#         y_indices, x_indices = np.where(labels == label)
#         limits[label] = (min(x_indices), max(x_indices))
#     return limits



# def split_letter(original_image, limits):
#     """
#     Split the original image into individual letters based on the provided limits.
#     """
#     letters = []
#     for label, (start_col, end_col) in limits.items():
#         letter_image = original_image[:, start_col:end_col]
#         letters.append(letter_image)
#     return letters


#directory = "./CNN_generated_dataset"
image_path = "test_plate.png"

#image_path = os.path.join(directory, filename)
gray_image = load_image(image_path)
edge_image = detect_contour(gray_image)
plot_detect_contour(gray_image, edge_image)

white_pixel_counts = count_white_pixels_per_column(edge_image)
plot_histogram(white_pixel_counts)

slices = slice_image_based_on_histogram(gray_image, white_pixel_counts)

for i, slice_img in enumerate(slices):
    plt.imshow(slice_img, cmap='gray')
    plt.title(f"Slice {i+1}")
    plt.show()

# labels = determine_number_of_letters(edge_image)
# limits = find_limits(labels)
# letters = split_letter(gray_image, limits)

# Vous pouvez maintenant traiter chaque lettre individuellement
# for letter in letters:
#     plt.imshow(letter, cmap='gray')
#     plt.show()
