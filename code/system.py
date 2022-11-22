"""
COM2004: Word Search Puzzle Solver Assignment

Author: Filip J. Cierkosz

Development date: 10/2022-12/2022
"""


import numpy as np
import scipy
from typing import List
from itertools import combinations
from utils import utils
from utils.utils import Puzzle


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
    Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    Takes the raw feature vectors and reduces them down to the required number of
    dimensions. Note, the `model` dictionary is provided as an argument so that
    you can pass information from the training stage, e.g. if using a dimensionality
    reduction technique that requires training, e.g. PCA.

    Args:
        data (np.ndarray) : The feature vectors to be reduced.
        model (dict) : The dictionary that stores the model data.

    Returns:
        np.ndarray : The appropriately reduced feature vectors.
    """
    # load mean to perform mean normalization
    mean_train = model["mean_train"]
    # load eiegenvectors to perform the pca dimensionality reduction
    # so project the data into the N principal component axes (linear transform)
    eigv_train = model["eigv_train"]
    reduced_data = np.dot((data - mean_train), eigv_train)
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    This is your classifier's training stage. You need to learn the model parameters
    from the training vectors and labels that are provided. The parameters of your
    trained model are then stored in the dictionary and returned. Note, the contents
    of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    The dummy implementation stores the labels and the dimensionally reduced training
    vectors. These are what you would need to store if using a non-parametric
    classifier such as a nearest neighbour or k-nearest neighbour classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    # MIGHT BE POSSIBLE TO INVOKE THE FEATURE SELECTION FROM THIS STAGE

    # basis for model setup
    model = {}
    # just the labels
    model["labels_train"] = labels_train.tolist()
    # compute the mean and store it in the model dictionary 
    model["mean_train"] = np.mean(fvectors_train)
    # construct covariance matrix from the training data 
    # then compute the first 2-21 eigenvectors (principal component axes)
    # as column vector in the matrix v. w is _ since it is never used
    cov_matrix = np.cov(fvectors_train, rowvar=0)
    dim = cov_matrix.shape[0]
    _, eigv = scipy.linalg.eigh(cov_matrix, eigvals=(dim - 40, dim - 1))
    eigv = np.fliplr(eigv)
    # call the new function for feature selection
    pca_data = np.dot((fvectors_train - np.mean(fvectors_train)), eigv)
    best_eigv_indices = select_pca_vectors(pca_data, model)
    # refine th eigenvector matrix of 40 into the best selected 20
    eigv = eigv[:, best_eigv_indices]
    # store eigenvectors in the model as well (top 20 now)
    model["eigv_train"] = eigv.tolist()
    # reduce the dimensions using the best 20 eigenvectors
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    # update the model
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def select_pca_vectors(pca_data, model):
    """
    find the best 20 eigenvectors out of all 40 computed in the train data
    """
    # prepare all combinations of class labels
    labels_list = np.array(model["labels_train"])
    labels_set = sorted(set(labels_list))
    labels_pairs = list(combinations(labels_set, 2))

    # dictionary to keep track of the most valuable
    # eigenvectors (out of all 40)
    weighed_eigv_dict = {k: 0 for k in range(pca_data.shape[1])}

    # test each possible pair of labels 
    for pair in labels_pairs:
        l1, l2 = pair
        l1_data = pca_data[labels_list == l1, :]
        l2_data = pca_data[labels_list == l2, :]
        # specify coefficient that increases the dictionary (penalty?)
        coeff = np.min([l1_data.shape[0], l2_data.shape[0]])

        if coeff > 10:
            # calc divergence and based on the coeff proceed furthermore
            try:
                div12 = calc_divergence(l1_data, l2_data)
                sorted_indexes = np.argsort(-div12)
                print(sorted_indexes)
                best_eigv_indices = sorted_indexes[0:8]
                # increase the dictionary where suitable 
                for i in best_eigv_indices:
                    weighed_eigv_dict[i] += coeff 
            except ValueError:
                continue
        else:
            continue
    
    # sort the dicitionary in descending order
    sorted_weighed_eigv_desc = dict(sorted(weighed_eigv_dict.items(), key=lambda item: item[1])[::-1])
    print(sorted_weighed_eigv_desc)
    # get the best 20 eiegenvectors
    selected_eigv_indices = list(sorted_weighed_eigv_desc.keys())[0:20]
    return selected_eigv_indices


def calc_divergence(class1, class2):
    """
    compute a vector of 1-D divergences between two classes

    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2

    returns: d12 - a vector of 1-D divergence scores

    CODE SOLUTION INSPIRED BY THE ONE PROVIDED FOR LAB 6
    """
    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * (m1 - m2) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)
    return d12


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
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
    # mdist = np.max(dist, axis=1)
    # assign labels for the test data
    test_labels = labels_train[nearest]
    return test_labels


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
