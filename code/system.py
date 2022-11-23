"""
COM2004: Word Search Puzzle Solver Assignment

Author: Filip J. Cierkosz

Development date: 10/2022-12/2022
"""


import numpy as np
import scipy
from typing import List
from collections import Counter
from itertools import combinations
from utils import utils
from utils.utils import Puzzle

# DEBUGGING - DELETE THIS LATER
import matplotlib.pyplot as plt
import matplotlib


# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """
    Extract raw feature vectors for each puzzle from images in the image_dir.

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    Args:
        image_dir (str) : Name of the directory where the puzzle images are stored.
        puzzle (dict) : Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray : The raw data matrix, i.e. rows of feature vectors.
    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """
    Perform dimensionality reduction on the set of feature vector down
    to specified N_DIMENSIONS (20) using the best 20 eigenvectors (principal
    component axes) that were selected during the training stage using PCA
    approach.

    In essence, the function takes raw feature vectors of a data set and 
    reduces them down to N_DIMENSIONS (20).

    Args:
        data (np.ndarray) : Feature vectors to be reduced.
        model (dict) : Dictionary that stores the essential model data from training.

    Returns:
        reduced_data (np.ndarray) : Feature vectors (input data) reduced to N_DIMENSIONS.
    """
    # Use mean to perform mean normalization, then use eigenvectors to perform 
    # the PCA dimensionality reduction, i.e. to project the data into the N 
    # principal component axes (it's a linear transform).
    mean_train = model["mean_train"]
    eigv_train = model["eigv_train"]
    reduced_data = np.dot((data - mean_train), eigv_train)
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """
    Perform the classifier's training stage by processing the training data. 
    Start by computing 40 eiegenvectors for 40 highest eigenvalues, then use them
    to perform a feature selection, i.e. to select the 20 most useful eiegenvectors
    calling a function dedicated for this task. Finally, after learning the model 
    parameters using the Prinicipal Component Analysis approach, store the valuable
    information (i.e. labels, mean, top eiegenvectors, reduced training feature vectors) 
    in a model dicionary in order to use it as a basis for the latter classification. 

    Args:
        fvectors_train (np.ndarray) : Training data feature vectors (stored as rows).
        labels_train (np.ndarray) : Labels corresponding to the training feature vectors.
    
    Returns:
        model (dict) : Dictionary that stores the model of the data that was learned 
                       during the training stage.
    """
    model = {}
    model["labels_train"] = labels_train.tolist()

    # PERFORM BINARIZATION???
    # fvectors_train = binarize_data(fvectors_train)

    # Compute the mean, as it will be later used for dimensionality reduction.
    model["mean_train"] = np.mean(fvectors_train)
    # Construct covariance matrix and use it to compute the 40 eigenvectors 
    # corresponding to the 40 highest eigenvalues.
    cov_matrix = np.cov(fvectors_train, rowvar=0)
    cov_dim = cov_matrix.shape[0]
    _, eigv = scipy.linalg.eigh(
        cov_matrix, eigvals=(cov_dim - 40, cov_dim - 1)
    )
    eigv = np.fliplr(eigv)
    # Perform mean normalization, currently for the 40 eiegenvectors.
    # Use it to perform feature selection, i.e. find 20 most useful eigv out of 40.
    pca_data = np.dot(
        (fvectors_train - np.mean(fvectors_train)), eigv
    )
    selected_eigv_indices = select_features_pca(pca_data, N_DIMENSIONS, model)
    # Select the best 20 eiegenvectors, out of 40 computed. Store them in model.
    eigv = eigv[:, selected_eigv_indices]
    model["eigv_train"] = eigv.tolist()
    # Reduce dimensions of the training set, which is based on top 20 eigv.
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def select_features_pca(pca_data: np.ndarray, N: int, model: dict) -> np.ndarray:
    """
    Perform feature selection for the computed principal component axes of the 
    training feature vectors. The selection process finds the N most useful 
    eigenvectors out of M eiegenvectors that were passed as pca_data (note: M > N). 

    The feature selection is based on the following approach:    
    1. Iterate through a list of all possible pairs of class labels in the data set.
    2. If the criteria for the minimum data frequency (MIN_DATA_FREQ) is met,
       then compute divergences on the current pair and select indices of the 
       8 highest scoring eigenvectors. Furthermore, increase the value of the 
       corresponding indices of the top scoring eigenvectors (i.e. of keys in 
       weigh_eigv_dict) by the value of the minimum data frequency for current pair.
    3. Sort the dictionary in descending order by the values and based on that select 
       the top N keys (i.e. eigenvector indices) after testing each possible pair.

    Args:
        pca_data (np.ndarray) : Set of M principal component axes computed for the
                                training set, out of which top N will be selected.
        N (int) : Number of features to be selected in feature selection process.
        model (dict) : Dictionary that stores useful information about training data.

    Returns:
        nbest_eigv_indices (np.ndarray) : Indices of the selected top N eiegenvectors.
    """
    MIN_DATA_FREQ = 10

    weigh_eigv_dict = {k: 0 for k in range(pca_data.shape[1])}
    labels_list = np.array(model["labels_train"])
    labels_set = sorted(set(labels_list))
    labels_pairs = list(combinations(labels_set, 2))

    for pair in labels_pairs:
        l1, l2 = pair
        l1_data = pca_data[labels_list == l1, :]
        l2_data = pca_data[labels_list == l2, :]
        curr_data_freq = np.min([l1_data.shape[0], l2_data.shape[0]])

        if curr_data_freq > MIN_DATA_FREQ:
            try:
                div12 = calc_divergence(l1_data, l2_data)
                sorted_indexes = np.argsort(-div12)
                best_eigv_keys = sorted_indexes[0:8]
                for key in best_eigv_keys:
                    weigh_eigv_dict[key] += curr_data_freq
            except ValueError:
                continue
    
    # Sort the dictionary in descending order. Retrieve the top N keys,
    # which denote the indices of the top N eiegenvectors.
    sorted_desc_weigh_eigv_dict = dict(
        sorted(weigh_eigv_dict.items(), key=lambda item: item[1])[::-1]
    )
    nbest_eigv_indices = list(sorted_desc_weigh_eigv_dict.keys())
    return nbest_eigv_indices[0:N]


def binarize_data(fvectors: np.ndarray) -> np.ndarray:
    """
    Data pre-processing method based on the concept of binarization.
    It means that a matrix of pixels of a grayscale image is converted into 
    a binarized matrix, where each pixel is either assigned to the max value
    in the dataset (i.e. the most white), or to the min value, which marks 
    the blackest pixel. The classification of a pixel as either black or white
    is based on the value of thresh, which is...

    Args:
        fvectors (np.ndarray): Unbinarized feature vectors, i.e. in grayscale format.

    Returns:
        fvectors_binarized (np.ndarray): Binarized feature vectors, i.e. each pixel
                                         is converted either to black (min value 
                                         in data), or white (max value in data).
    """
    # matplotlib.use("TkAgg")
    # imgold = fvectors[25, :].reshape(20, 20)
    # plt.imshow(imgold, cmap="gray")
    # plt.show()
    for y in range(fvectors.shape[0]):
        thresh = np.mean(fvectors[y, :]) * 1.5
        black = np.min(fvectors[y, :])
        white = np.max(fvectors[y, :])
        for x in range(fvectors.shape[1]):
            if fvectors[y, x] > thresh:
                fvectors[y, x] = white
            else:
                fvectors[y, x] = black
    # imgnew = fvectors[25, :].reshape(20, 20)
    # plt.imshow(imgnew, cmap="gray")
    # plt.show()
    return fvectors


def calc_divergence(fvectors_class1: np.ndarray, fvectors_class2: np.ndarray) -> np.ndarray:
    """
    Compute the vector of 1D divergence between vectors of two classes.
    Proceed by computing means and variances of each of the two classes.
    Then, use them in the formula to compute the 1D divergence.

    Args:
        fvectors_class1 (np.ndarray) : Feature vectors of class 1 (each row is a sample).
        fvectors_class2 (np.ndarray) : Feature vectors of class 2 (each row is a sample).

    Returns:
        div12 (np.ndarray) : Vector of 1D divergence scores between two classes.

    NOTE: Code implementation inspired by the one provided by 
          COM2004 academic staff in Lab Sheet 6.
    """
    m1 = np.mean(fvectors_class1, axis=0)
    m2 = np.mean(fvectors_class2, axis=0)
    v1 = np.var(fvectors_class1, axis=0)
    v2 = np.var(fvectors_class2, axis=0)
    div12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)
    return div12


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """
    Docs here...
    """
    """Dummy implementation of classify squares.

    This is the classification stage. You are passed a list of unlabelled feature
    vectors and the model parameters learn during the training stage. You need to
    classify each feature vector and return a list of labels.

    Args:
        fvectors_test (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # perform the nn classification 
    x = np.dot(fvectors_test, fvectors_train.transpose())
    mod_test = np.sqrt(np.sum(fvectors_test * fvectors_test, axis=1))
    mod_train = np.sqrt(np.sum(fvectors_train * fvectors_train, axis=1))
    # calc cosine distance
    dist = x / np.outer(mod_test, mod_train.transpose())
    nearest = np.argmax(dist, axis=1)

    # trying to consider k nearest neighbours
    K = 3 # 9 gives 57% for low quality
    k_nearest = np.argsort((-dist), axis=1)[:, 0:K]
    # map sample to an appropriate label
    nearest_beta = []

    # check the labels
    # weighed knn
    for kn in k_nearest:
        # print(labels_train[kn[0]])
        k_labels = labels_train[kn]
        labels_counted = Counter(k_labels)
        # print(labels_counted)

        # if max counted has several occurences then select the best single one
        # even if all are counted as one, we will consider the first element 
        currbest = max(labels_counted, key=labels_counted.get)
        nearest_beta.append(currbest)

    # print(nearest_beta)

    # assign labels for the test data
    # test_labels = labels_train[nearest]
    return nearest_beta


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Dummy implementation of find_words.

    REWRITE THIS FUNCTION AND THIS DOCSTRING

    This function searches for the words in the grid of classified letter labels.
    You are passed the letter labels as a 2-D array and a list of words to search for.
    You need to return a position for each word. The word position should be
    represented as tuples of the form (start_row, start_col, end_row, end_col).

    Note, the model dict that was learnt during training has also been passed to this
    function. Most simple implementations will not need to use this but it is provided
    in case you have ideas that need it.

    In the dummy implementation, the position (0, 0, 1, 1) is returned for every word.

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    # NEED TO DIVIDE THIS PART INTO 3 FUNCTIONS: row, column, diagonal for search

    return [(0, 0, 1, 1)] * len(words)
