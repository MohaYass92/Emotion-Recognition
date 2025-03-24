from tqdm import tqdm
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler

def extract_hog_features(image):
    if len(image.shape) == 3:
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def extract_features_from_generator(generator):
    """ Extract HOG features from all images in a generator with a progress bar. """
    features_list = []
    labels_list = []

    total_batches = len(generator)  # Get total number of batches
    with tqdm(total=total_batches, desc="Extracting Features", unit="batch") as pbar:
        for batch in generator:
            images, labels = batch
            for image in images:
                features = extract_hog_features(image)
                features_list.append(features)
            labels_list.extend(np.argmax(labels, axis=1))
            pbar.update(1)  # Update progress bar after each batch

    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    scaler = StandardScaler()
    features_array = scaler.fit_transform(features_array)

    return features_array, labels_array
