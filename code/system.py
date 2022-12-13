"""
COM2004: Word Search Puzzle Solver Assignment

Author: Filip J. Cierkosz

Development date: 10/2022 -- 12/2022
"""


from typing import List
from itertools import combinations, product
import numpy as np
import scipy
from scipy.spatial.distance import cdist, hamming
from utils import utils
from utils.utils import Puzzle


# The required maximum number of dimensions for reducedfeature vectors.
N_DIMENSIONS = 20

# Minimum data frequency of a particular class label in the dataset
# to be considered in the feature selection process for PCA.
MIN_DATA_FREQ = 10

# The K value used in K-Nearest-Neigbor classification process.
KNN_VAL = 9


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
    Perform dimensionality reduction on the set of feature vectors down
    to specified N dimensions (i.e. 20) using the best N (i.e. 20 again)
    principal component axes (eigenvectors) that were selected during
    the training stage for the PCA approach. In essence, the function
    converts the raw feature vectors of M dimensions down to N dimensions.

    Args:
        data (np.ndarray) : Raw feature vectors to be reduced.
        model (dict) : Dictionary that stores the useful model data saved from training.

    Returns:
        reduced_data (np.ndarray) : Feature vectors reduced to N dimensions.
    """
    mean_train = model["mean_train"]
    eigv_train = model["eigv_train"]
    reduced_data = np.dot((data - mean_train), eigv_train)
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """
    Perform the classifier's training stage by processing the training data.
    Start by initially computing 40 eigenvectors for 40 highest eigenvalues to
    then use them in feature selection stage, which is used to select the N (i.e. 20)
    most useful eigenvectors out of the 40 computed. Finally, after learning the model
    parameters using the PCA approach, the most valuable information is stored (e.g. mean,
    selected eiegenvectors) in the model dicionary, which is later used in classification.

    Args:
        fvectors_train (np.ndarray) : Training data feature vectors (stored as rows).
        labels_train (np.ndarray) : Labels corresponding to the training feature vectors.

    Returns:
        model (dict) : Dictionary that stores the model of the data that was learned
                       during the training stage.
    """
    model = {}
    model["labels_train"] = labels_train.tolist()
    model["mean_train"] = np.mean(fvectors_train)

    # Compute covariance matrix and use it to calculate 40
    # eigenvectors corresponding to the 40 greatest eigenvalues.
    cov_matrix = np.cov(fvectors_train, rowvar=0)
    n_cov = cov_matrix.shape[0]
    _, eigv = scipy.linalg.eigh(cov_matrix, eigvals=(n_cov - 40, n_cov - 1))
    eigv = np.fliplr(eigv)

    # Apply mean normalization for the 40 eiegenvectors, and then perform
    # feature selection, where the best N (20) PCAs (eigenvectors) are selected.
    pca_data = np.dot((fvectors_train - np.mean(fvectors_train)), eigv)
    selected_eigv_indices = select_features_pca(pca_data, model)
    eigv = eigv[:, selected_eigv_indices]
    model["eigv_train"] = eigv.tolist()

    # Reduce dimensions of the training set (based on the N selected PCAs).
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def select_features_pca(pca_data: np.ndarray, model: dict) -> np.ndarray:
    """
    Perform feature selection for the computed principal component axes of the
    training feature vectors. The selection process finds the N most useful
    eigenvectors out of M eiegenvectors that were passed as pca_data (note: M >= N).

    The feature selection approach is performed in 3 main stages:
    1. Iterate through the list of all possible pairs of class labels in the data set.
    2. If the criteria for the minimum data frequency (MIN_DATA_FREQ) for both labels in
       pair is met, then compute divergences on the current pair and select indices of the
       3 highest scoring eigenvectors. Furthermore, increase the value of the
       corresponding indices of the top scoring eigenvectors (i.e. of keys in
       weigh_eigv_dict) by the value of the data frequency of a current pair.
    3. Sort the dictionary in descending order by the values and based on that select
       the top N keys, which correspond to the selected eigenvectors' indices.

    Args:
        pca_data (np.ndarray) : Set of M principal component axes computed for the
                                training set, out of which the top N will be selected.
        model (dict) : Dictionary that stores useful information about training data.

    Returns:
        nbest_eigv_indices (np.ndarray) : Indices of the selected top N eiegenvectors.
    """
    labels_list = np.array(model["labels_train"])
    labels_set = sorted(set(labels_list))
    labels_pairs = list(combinations(labels_set, 2))
    weigh_eigv_dict = {k: 0 for k in range(pca_data.shape[1])}

    for (label1, label2) in labels_pairs:
        l1_data = pca_data[labels_list == label1, :]
        l2_data = pca_data[labels_list == label2, :]
        curr_data_freq = np.min([l1_data.shape[0], l2_data.shape[0]])

        # Rank current pair's eigenvectors using the 1D divergence matrix
        # if the data frequency criteria is met by a particular pair.
        if curr_data_freq > MIN_DATA_FREQ:
            try:
                div12 = calc_divergence(l1_data, l2_data)
                best_eigv_keys = np.argsort(-div12)[0:3]
                for key in best_eigv_keys:
                    weigh_eigv_dict[key] += curr_data_freq
            except ValueError:
                continue

    # Sort the dictionary in descending order. Retrieve the top N keys,
    # which denote the indices of the top N eiegenvectors.
    sorted_desc_weigh_eigv_dict = dict(
        sorted(weigh_eigv_dict.items(), key=lambda item: item[1])[::-1]
    )
    return list(sorted_desc_weigh_eigv_dict.keys())[0:N_DIMENSIONS]


def calc_divergence(fvectors_class1: np.ndarray, fvectors_class2: np.ndarray) -> np.ndarray:
    """
    Compute the vector of 1D divergence between vectors of two classes.
    Proceed by computing means and variances of each of the two classes.
    Then, use them in the formula to compute the 1D divergence.

    Args:
        fvectors_class1 (np.ndarray) : Feature vectors of class 1 (a row is a sample).
        fvectors_class2 (np.ndarray) : Feature vectors of class 2 (a row is a sample).

    Returns:
        div12 (np.ndarray) : Vector of 1D divergence scores between the two tested classes.

    NOTE: Code implementation inspired by the one provided by
          COM2004 academic staff in Lab Sheet 6.
    """
    mean1 = np.mean(fvectors_class1, axis=0)
    mean2 = np.mean(fvectors_class2, axis=0)
    var1 = np.var(fvectors_class1, axis=0)
    var2 = np.var(fvectors_class2, axis=0)
    div12 = 0.5 * (var1 / var2 + var2 / var1 - 2) +\
            0.5 * (mean1 - mean2) * (mean1 - mean2) * (1.0 / var1 + 1.0 / var2)
    return div12


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """
    Perform classification for the reduced test feature vectors using the K-Nearest-
    -Neighbor approach with weights. Use the scipy-provided "correlation" distance metric
    to compare the feature vectors as accurately as possible and find k closest distances
    and consequently labels. The introduced weights are found by calculating the inverse
    square of distance for each of the k closest classes. The label with highest
    weight is selected as the best choice for the classification of a particular sample.

    Args:
        fvectors_test (np.ndarray) : Feature vectors of the test dataset that will
                                     be classified (feature vectors stored as rows).
        model (dict) : Dictionary that stores the essential information learned
                       during the training stage (e.g. reduced feature vectors).

    Returns:
        test_labels (List[str]): A list of classified class labels; one class
                                 label (alphabet letter 'A'-'Z') per feature vector.
    """
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    dists = cdist(fvectors_test, fvectors_train, "correlation")
    k_nearest_dists = np.sort((dists), axis=1)[:, 0:KNN_VAL]
    k_nearest_labels = [labels_train[i] for i in np.argsort((dists), axis=1)[:, 0:KNN_VAL]]
    test_labels = []

    # Perform weighted KNN part of the algorithm.
    for i in range(k_nearest_dists.shape[0]):
        weights_dict = {}

        for label, dist in zip(k_nearest_labels[i], k_nearest_dists[i]):
            weight = 1 / (dist * dist)

            if not label in weights_dict:
                weights_dict[label] = weight
            else:
                weights_dict[label] +=  weight

        best_label = max(weights_dict, key=weights_dict.get)
        test_labels.append(best_label)

    return test_labels


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """
    Search for words in the puzzle grid of the classified letter labels. For each
    word, use the function that searches for the word in all the directions at once
    and finds coordinates of either the closest or exact match of a particular word.

    Args:
        labels (np.ndarray) : 2D array that stores a classified letter in each
                              square of the wordsearch puzzle grid.
        words (list[str]) : List of words to be found in the word-search puzzle.
        model (dict) : The essential model parameters learned during training.

    Returns:
        words_pos (list[tuple]): List of four-element tuples indicating each
                                 word's start and end position in the puzzle grid.
    """
    words = [w.upper() for w in words]
    words_pos = []

    for word in words:
        pos = search_word_pos(word, labels)
        words_pos.append(pos)

    return words_pos


def search_word_pos(word: str, label_grid: np.ndarray) -> tuple:
    """
    Search for a targetted word in all directions (i.e. row, column, diagonal).
    The algortihm starts by computing all possible starting and ending positions
    of a word in the puzzle grid (label grid) considering the length of the
    targetted word. Then, each guess is built out of the letters found in the
    label grid and passed to function which returns closest match of the target.

    Args:
        word (str) : target word to be found in the label grid.
        label_grid (np.ndarray) : the grid of classified labels that construct the
                                  puzzle grid for the word search game.

    Returns:
        cm_pos (tuple) : the position in the label grid of a string that is the closest
                         match to the target word; the position is stored in a tuple
                         in the format (r1, c1, r2, c2), where r1 and c1 determine the
                         position (in row and column) of the first letter, and r2 and c2
                         determine th position of the ending letter of the word.
    """
    w_len = len(word)
    # Compute all possible starting and ending indices of any string in a label
    # grid for both row and column of the label grid using Cartesian Product.
    irows = list(product(list(range(label_grid.shape[0])), repeat=2))
    icols = list(product(list(range(label_grid.shape[1])), repeat=2))

    # Compute all possible start and end positions of the searched word in the grid
    # considering all directions at once (i.e. diagonal, row, column) and matching the word length.
    expected_word_pos = [
        (r1, c1, r2, c2) for c1, c2 in icols for r1, r2 in irows
            if (abs(r1 - r2) == (w_len - 1) and abs(c1 - c2) == (w_len - 1)) or
               (abs(r1 - r2) == (w_len - 1) and c1 == c2) or
               (abs(c1 - c2) == (w_len - 1) and r1 == r2)
    ]

    # Create all possible word guesses from subsequent elements in the grid.
    wmatch_pos_dict = {}
    for (r_start, c_start, r_end, c_end) in expected_word_pos:
        irow = setup_coords(r_start, r_end, w_len)
        icol = setup_coords(c_start, c_end, w_len)
        w_guess = "".join([label_grid[r, c] for r, c in zip(irow, icol)])
        wmatch_pos_dict[w_guess] = (r_start, c_start, r_end, c_end)

    closest_match = find_closest_match(word, wmatch_pos_dict.keys())
    return wmatch_pos_dict[closest_match]


def setup_coords(coor_start: int, coor_end: int, w_len: int) -> List[int]:
    """
    Set up list of start to end coordinates for a word guess' row, or column,
    in the label grid. Consideration that they might be in descending order
    (e.g. 5 is starting, 1 is ending), or all the sam values (e.g. in case
    column coords are unchanged, and only row ) has been also handled.

    Args:
        coor_start (int) : start coordinate in a row/column.
        coor_end (int) : end coordinate in a row/column.
        w_len (int) : Length of the searched word.

    Returns:
        icoords (List[int]) :
    """
    icoords = []
    if coor_start < coor_end:
        icoords = list(range(coor_start, (coor_end + 1)))
    elif coor_start > coor_end:
        icoords = list(range(coor_start, (coor_end - 1), -1))
    else:
        icoords = [coor_start] * w_len
    return icoords


def find_closest_match(searched_word: str, word_guesses: List[str]) -> str:
    """
    Find the closest match for a target word by comparing how many letters in each
    word guess from the list are the same as in the searched word and calculating
    the hamming distances accordingly. The guess with the lowest hamming distance
    is returned. In case of more than one guesses having the same hamming, then
    try to find the first one matching the first and last letter in the target word.
    If not possible, then return the last element of the list by default.

    Args:
        searched_word (str) : Target word to be matched.
        word_guesses List[str] : List of guessed words based on sequences of letters
                                 in the label grid in all directions; each guess
                                 match the length of the target word as well.

    Returns:
        closest_match : closest match to the searched word found by the max score.
    """
    # Build the dictionary of guesses with referring hamming distances.
    wguess_hamming_dict = {wg: hamming(list(wg), list(searched_word)) for wg in word_guesses}
    min_hamming_dist = min(wguess_hamming_dict.values())
    closest_matches = [wg for wg, d in wguess_hamming_dict.items() if d == min_hamming_dist]

    if len(closest_matches) > 1:
        for match in closest_matches:
            if match[0] == searched_word[0] and match[-1] == searched_word[-1]:
                return match

    return closest_matches[-1]
