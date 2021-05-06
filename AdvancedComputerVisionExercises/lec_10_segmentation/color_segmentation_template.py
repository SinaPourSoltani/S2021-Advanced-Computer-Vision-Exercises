import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from copy import deepcopy

import sys
sys.path.append("../")
from lib.visualization.image import put_text


def load_image(image_file):
    """
    Loads the image and converts to LAB color space. Return the flatten LAB image and the original image dimensions

    Parameters
    ----------
    image_file (str): The file path to the image

    Returns
    -------
    image_vector (ndarray): The flatten image in LAB color space
    dims (tuple): The original image dimensions
    """
    # Read the image
    # Convert to LAB color space
    # Save original dimensions for later use
    # Flatten image along height and width (not depth)
    # Return flattened image and dimensions
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    dim = image.shape

    data = np.array(image)
    image_vector = data.reshape((-1, 3))
    print(image_vector.shape, dim)
    return image_vector, dim


def quantize_image(image_vector, n_clusters):
    """
    Quantize the image

    Parameters
    ----------
    image_vector (ndarray): The image vector
    k (int): Number of clusters

    Returns
    -------
    quantize_image_vector (ndarray): The quantize image vector
    """
    # Create kmeans object with k clusters
    # Fit the image
    # Index the kmeans.cluster_centers_ with the predicted labels in order to quantize the image
    # Return the quantized image
    kmeans = MiniBatchKMeans(n_clusters).fit(image_vector)
    quantize_image_vector = kmeans.fit_predict(image_vector)

    quantized_image = np.ndarray((image_vector.shape), dtype='uint8')
    quantized_image = kmeans.cluster_centers_[quantize_image_vector]

    return quantized_image


def show_quantization(image_vector, shape, n_clusters):
    """
    Show's the quantized image

    Parameters
    ----------
    image_vector (ndarray): The image vector
    shape (tuple): The original image dimensions
    k (int): Number of clusters
    """
    # Reshape the image to original dimensions
    # Convert to BGR color space
    # Show quantized image
    image_vector = image_vector.reshape(shape)

    image_vector = cv2.cvtColor(image_vector.astype('uint8'), cv2.COLOR_LAB2BGR)
    cv2.imshow("quantized " + str(n_clusters), image_vector)
    cv2.waitKey()


if __name__ == "__main__":
    image_file = "../data/color_segmentation/image.jpg"
    # Load the image and the get original image dimensions
    image, orig_shape = load_image(image_file)

    cluster_range = range(1,10)  # The range of clusters to test

    for k in reversed(cluster_range):
        # Quantize the image with k clusters
        quant = quantize_image(image, k)
        # Show the quantize image
        show_quantization(quant, orig_shape, k)
