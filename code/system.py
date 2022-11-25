"""
COM2004: Word Search Puzzle Solver Assignment

Author: Filip J. Cierkosz

Development date: 10/2022-12/2022
"""


import numpy as np
import scipy
from typing import List
from collections import Counter
from itertools import combinations, permutations
from utils import utils
from utils.utils import Puzzle


# The required maximum number of dimensions for feature vectors.
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
    to specified N_DIMENSIONS (i.e. 20) using the best N (i.e. 20 again) 
    eigenvectors (principal component axes) that were selected during 
    the training stage using Principal Component Analysis approach.

    Essentially, the function takes raw feature vectors of M dimensions and
    reduces them down to the specified N_DIMENSIONS (i.e. 20).

    Args:
        data (np.ndarray) : Raw feature vectors to be reduced.
        model (dict) : Dictionary that stores the useful model data from training (e.g mean).

    Returns:
        reduced_data (np.ndarray) : Feature vectors reduced to N_DIMENSIONS.
    """
    # Perform mean normalization, then use eigenvectors to to project 
    # the data into the N principal component axes (linear transform).
    mean_train = model["mean_train"]
    eigv_train = model["eigv_train"]
    reduced_data = np.dot((data - mean_train), eigv_train)
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """
    Perform the classifier's training stage by processing the training data. 
    Start by computing 40 eigenvectors for 40 highest eigenvalues, then use them
    to perform a feature selection, i.e. to select the 20 most useful eiegenvectors
    calling a function dedicated for this task. Finally, after learning the model 
    parameters using the Prinicipal Component Analysis approach, store the most valuable
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
    model["mean_train"] = np.mean(fvectors_train)

    # Compute covariance matrix and use it to calculate 40 
    # eigenvectors corresponding to the 40 greatest eigenvalues.
    cov_matrix = np.cov(fvectors_train, rowvar=0)
    N_COV = cov_matrix.shape[0]
    _, eigv = scipy.linalg.eigh(
        cov_matrix, eigvals=(N_COV - 40, N_COV - 1)
    )
    eigv = np.fliplr(eigv)

    # Apply mean normalization for the 40 eiegenvectors, and then perform
    # feature selection where the best N (20) eigenvectors are selected.
    pca_data = np.dot(
        (fvectors_train - np.mean(fvectors_train)), eigv
    )
    selected_eigv_indices = select_features_pca(
        pca_data, N_DIMENSIONS, model
    )
    eigv = eigv[:, selected_eigv_indices]
    model["eigv_train"] = eigv.tolist()

    # Reduce dimensions of the training set (based on the top N eigv).
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
    return model


def binarize_data(fvectors: np.ndarray) -> np.ndarray:
    """
    USEFUL CONCEPT IN CV - BUT NOT USEFUL IN TRAINING STAGE!

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
    imgold = fvectors[25, :].reshape(20, 20)
    # plt.imshow(imgold, cmap="gray")
    # plt.show()
    for y in range(fvectors.shape[0]):
        thresh = np.mean(fvectors[y, :]) * 1.5
        black = np.min(fvectors[y, :])
        white = np.max(fvectors[y, :])
        fvectors[y, :] = np.where(
            fvectors[y, :] > thresh, white, fvectors[y, :]
        )
        fvectors[y, :] = np.where(
            fvectors[y, :] <= thresh, black, fvectors[y, :]
        )
    imgnew = fvectors[25, :].reshape(20, 20)
    # plt.imshow(imgnew, cmap="gray")
    # plt.show()
    return fvectors


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

    # Rank each pair's eigenvectors using the 1D divergence matrix
    # if the data frequency criteria is met by a pair.
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


def calc_divergence(fvectors_class1: np.ndarray, fvectors_class2: np.ndarray) -> np.ndarray:
    """
    Compute the vector of 1D divergence between vectors of two classes.
    Proceed by computing means and variances of each of the two classes.
    Then, use them in the formula to compute the 1D divergence.

    Args:
        fvectors_class1 (np.ndarray) : Feature vectors of class 1 (a row is a sample).
        fvectors_class2 (np.ndarray) : Feature vectors of class 2 (a row is a sample).

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
    Classify unlabelled feature vectors... from the test data

    in k selection: if max counted has several occurences then select the 
    best single one even if all are counted as one, the first element will be considered

    Args:
        fvectors_test (np.ndarray) : Feature vectors of the test dataset that will 
                                     be classified (feature vectors stored as rows).
        model (dict) : Dictionary that stores the essential information learned
                       during the training stage (e.g. reduced feature vectors).

    Returns:
        test_labels (List[str]): A list of classified class labels; one class 
                                 label (alphabet letter 'A'-'Z') per feature vector.
    """
    # SIGNAL TO NOISE???
    # mean = np.mean(fvectors_test, axis=1)
    # std = np.std(fvectors_test, axis=1)
    # signal_to_noise = np.where(std == 0, 0, mean / std)
    # print(np.mean(signal_to_noise))

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    K = 3 # int(math.sqrt(fvectors_test.shape[0])) # 9 gives 57% for low quality

    x = np.dot(fvectors_test, fvectors_train.transpose())
    mod_test = np.sqrt(
        np.sum(fvectors_test * fvectors_test, axis=1)
    )
    mod_train = np.sqrt(
        np.sum(fvectors_train * fvectors_train, axis=1)
    )
    # Calculate the cosine distance.
    dist = x / np.outer(mod_test, mod_train.transpose())
    # Select K-nearest neighbors (found by selecting 5 smallest distances for each sample). 
    k_nearest_indices = np.argsort((-dist), axis=1)[:, 0:K]
    # k_nearest_vals = np.sort((-dist), axis=1)[:, 0:K]
    test_labels = []

    # WEIGHTED KNN??
    for i in k_nearest_indices:
        k_labels = labels_train[i]
        labels_counted = Counter(k_labels)
        kn_best = max(labels_counted, key=labels_counted.get)
        test_labels.append(kn_best)

    return test_labels


def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """
    Search for words in the grid of classified letter labels. Letter labels
    are passed in 2D array. You should return (start_row, start_col, end_row, end_col)
    for each word

    Might need the model dict?

    Args:
        labels (np.ndarray) : 2D array that stores the character in each square 
                              of the wordsearch puzzle.
        words (list[str]) : A list of words to find in the wordsearch puzzle.
        model (dict) : The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """
    words = [w.upper() for w in words]
    words_pos = []
    wpos_score_dict = {}

    # search for words using separate approaches: horizontal, vertical, diagonal
    # select the one with the highest score
    for word in words:
        # get best match for row
        r_pos, r_score = search_rows(word, labels)
        # get best match for column
        c_pos, c_score = search_cols(word, labels)
        # wpos_score_dict[r_pos] = r_score
        # wpos_score_dict[c_pos] = c_score
        # get best match for diagonal
        # d_pos, d_score = search_diag(word, labels)

        if r_score > c_score:
            words_pos.append(r_pos)
        else:
            words_pos.append(c_pos)

        # words_pos.append(r_pos)
    print(words_pos)
    print('here')

    return words_pos


def find_closest_match(searched_word, word_guesses):
    """

    find the closest match for a word

    word : word to be matched
    word_guesses : list of guessed words that are matching the length of word

    closest_match : closest match of the searched word
    score : 
    """
    # build a dictionary to rank each guess
    rank_wguess_dict = {}

    # rank each guess checking how many letters are correct
    for wg in word_guesses:
        wg_score = sum(wg[i] == searched_word[i] for i in range(len(wg)))
        rank_wguess_dict[wg] = wg_score
    
    closest_match = max(rank_wguess_dict, key=rank_wguess_dict.get)
    cm_score = rank_wguess_dict[closest_match]
    return closest_match, cm_score


def search_rows(word: str, label_grid: np.ndarray): # -> tuple and int
    """
    Search for a word through each and every row.
    """
    nrow, ncol = label_grid.shape
    # all possible indices of columns in the grid.
    icols = [icol for icol in range(ncol + 1)]
    # compute all possible positions of the input word in the rows of
    # the label grid that match the length of the matched word (n)
    expected_word_indices = [
        (r, c1, r, c2) for r in range(nrow) for c1, c2 in combinations(icols, 2) 
            if abs(c1 - c2) == len(word)
    ]
    # build each possible string and map it with its position in the grid
    wmatch_pos_dict = {
        "".join(label_grid[r, c1:c2]): (r, c1, r, (c2 - 1)) 
            for r, c1, r, c2 in expected_word_indices
    }
    # build each possible reverse and map its position in the grid
    wmatch_rev_pos_dict = {
        "".join(label_grid[r, c1:c2][::-1]): (r, (c2 - 1), r, c1) 
            for r, c1, r, c2 in expected_word_indices
    }
    # concat both dictionaries to have a full list of all guesses
    # any repetetions of guesses are removed at this stage 
    # i.e. only one unique combination considered
    wmatch_pos_dict.update(wmatch_rev_pos_dict)
    # find the closest match among all
    closest_match, cm_score = find_closest_match(word, wmatch_pos_dict.keys())
    cm_pos = wmatch_pos_dict[closest_match]
    return cm_pos, cm_score


def search_cols(word: str, label_grid: np.ndarray): # tuple and int
    """
    Search for a word through all columns of the puzzle grid of labels.
    """
    nrow, ncol = label_grid.shape
    # all possible indices of rows in the grid.
    irows = [irow for irow in range(nrow + 1)]
    # compute all possible positions of the input word in the rows of
    # the label grid that match the length of the matched word (n)
    expected_word_indices = [
        (r1, c, r2, c) for c in range(ncol) for r1, r2 in combinations(irows, 2) 
            if abs(r1 - r2) == len(word)
    ]
    # build each possible string and map it with its position in the grid
    wmatch_pos_dict = {
        "".join(label_grid[r1:r2, c]): (r1, c, (r2 - 1), c) 
            for r1, c, r2, c in expected_word_indices
    }
    # build each possible reverse and map its position in the grid
    wmatch_rev_pos_dict = {
        "".join(label_grid[r1:r2, c][::-1]): ((r2 - 1), c, r1, c) 
            for r1, c, r2, c in expected_word_indices
    }
    # concat both dictionaries to have a full list of all guesses
    # any repetetions of guesses are removed at this stage 
    # i.e. only one unique combination considered
    wmatch_pos_dict.update(wmatch_rev_pos_dict)
    # find the closest match among all
    closest_match, cm_score = find_closest_match(word, wmatch_pos_dict.keys())
    cm_pos = wmatch_pos_dict[closest_match]
    return cm_pos, cm_score


def search_diag(word: str, label_grid: np.ndarray):
    """
    Search through diagonals (tough).
    """
    pass



# ARCHIVED
# def search_rows(word, label_grid):
#     """
#     Search for the current word through rows of the grid.
#     """
#     for r in range(label_grid.shape[0]):
#         r_joined = ''.join(label_grid[r, :])
#         word_match = r_joined.find(word)
#         word_rev_match = r_joined.find(word[::-1])

#         if word_match != -1:
#             c1 = word_match
#             c2 = word_match + (len(word) - 1)
#             return (r, c1, r, c2)
#         elif word_rev_match != -1:
#             c1 = word_rev_match + (len(word) - 1)
#             c2 = word_rev_match
#             return (r, c1, r, c2)

#     return (0,0,0,0)


# def search_cols(word, label_grid):
#     """
#     Search for the current word through columns of the grid.
#     """
#     for c in range(label_grid.shape[1]):
#         c_joined = ''.join(label_grid[:, c])

#         word_match = c_joined.find(word)
#         word_rev_match = c_joined.find(word[::-1])

#         # in case there is a palindrome, it does not matter which one is returned...
#         if word_match != -1:
#             r1 = word_match
#             r2 = word_match + (len(word) - 1)
#             return (r1, c, r2, c)
#         elif word_rev_match != -1:
#             r1 = word_rev_match + (len(word) - 1)
#             r2 = word_rev_match
#             return (r1, c, r2, c)

#     return (0,0,0,0)
