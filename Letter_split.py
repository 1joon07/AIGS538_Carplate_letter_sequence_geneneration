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
    _, otsu_thresh = cv2.threshold(gray_image, 0, 255,  cv2.THRESH_OTSU)
    return otsu_thresh
    
def detect_contour_canny(gray_image, lower_threshold = 100, upper_threshold = 255):
    canny_edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)
    _, edges = cv2.threshold(canny_edges, 0, 255, cv2.THRESH_BINARY)
    return edges

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

def count_black_pixels_per_column(image):
    """
    Count the number of black pixels in each column of the image.( this histogram will be used to split letters)
    """
    height, width = image.shape
    black_pixel_counts = []
    for col in range(width):
        column = image[:, col]
        black_pixels = np.sum(column < 20)
        black_pixel_counts.append(black_pixels)

    return black_pixel_counts

def plot_histogram(white_pixel_counts):
    plt.bar(range(len(white_pixel_counts)), white_pixel_counts)
    plt.xlabel('Column Index')
    plt.ylabel('Pixel Count')
    plt.title('Pixel Counts Per Column')
    plt.show()


# def count_white_pixels_per_column(image):
#     """
#     Count the number of black pixels in each column of the image.( this histogram will be used to split letters)
#     """
#     height, width = image.shape
#     white_pixel_counts = []
#     for col in range(width):
#         column = image[:, col]
#         white_pixels = np.sum(column > 200)
#         white_pixel_counts.append(white_pixels)

#     return white_pixel_counts


def slice_image_based_on_histogram(gray_image, black_pixel_counts, threshold=5, threshold_columns=10):
    """
    Slice the image into smaller images based on the histogram of black pixel counts.
    Each slice must be at least 'threshold_columns' wide. The decision to start or end a slice
    is based on the median count of the current, the  previous, and the  next columns, if they exist.
    """
    def median_count(idx):  # median to avoid outliner 
        counts = []
        for i in range(idx - 1, idx + 2):
            if 0 <= i < len(black_pixel_counts):
                counts.append(black_pixel_counts[i])
        return sorted(counts)[len(counts) // 2]  

    start_idx = None
    slices = []
    for idx in range(len(black_pixel_counts)):
        med_count = median_count(idx)
        if med_count >= threshold and start_idx is None:
            start_idx = idx
        elif start_idx is not None and (med_count < threshold or idx == len(black_pixel_counts) - 1):
            if idx - start_idx >= threshold_columns:
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


def pipeline_split_letter_wpath(image_path):
    '''
        Function to split letter.
        Input : path to the original
        Output : a list of image which contains 1 characters
    '''
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    otsu_image = detect_contour(gray_image)
    # edge_image = detect_contour_canny(gray_image)
    black_pixel_counts = count_black_pixels_per_column(otsu_image)
    # white_pixel_counts = count_white_pixels_per_column(edge_image)
    slices = slice_image_based_on_histogram(gray_image, black_pixel_counts)
    return slices

def pipeline_split_letter_wimage(image):
    '''
        Function to split letter.
        Input : the original image ( RGB )
        Output : a list of image which contains 1 characters
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = detect_contour(gray_image)
    black_pixel_counts = count_black_pixels_per_column(edge_image)
    slices = slice_image_based_on_histogram(gray_image, black_pixel_counts)
    return slices

if __name__ == '__main__':
    # directory = "./CNN_generated_dataset2"
    # filename = "Plate_10.png"

    # image_path = os.path.join(directory, filename)
    # image_path = "./test_plate.png"
    # image_path = "result_2.png"
    image_path = "result_1.png"
    gray_image = load_image(image_path)
    otsu_image = detect_contour(gray_image)
    edge_image = detect_contour_canny(gray_image)

    plot_detect_contour(gray_image, otsu_image)
    # plot_detect_contour(gray_image, edge_image)

    black_pixel_counts = count_black_pixels_per_column(otsu_image)
    plot_histogram(black_pixel_counts)

    # white_pixel_counts = count_white_pixels_per_column(edge_image)
    # plot_histogram(white_pixel_counts)

    slices = slice_image_based_on_histogram(gray_image, black_pixel_counts)
    print(len(slices))
    for i, slice_img in enumerate(slices):
        plt.imshow(slice_img, cmap='gray')
        plt.title(f"Slice {i+1}")
        plt.show()

# labels = determine_number_of_letters(edge_image)
# limits = find_limits(labels)
# letters = split_letter(gray_image, limits)
# for letter in letters:
#     plt.imshow(letter, cmap='gray')
#     plt.show()
