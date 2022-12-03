"""
COM2004: Word Search Puzzle Solver Assignment

Author: Filip J. Cierkosz

Development date: 10/2022-12/2022
"""


from typing import List
from collections import Counter
from itertools import combinations, product
import numpy as np
import scipy
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
    principal component axes (eigenvetors) that were selected during
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

    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    K = 9 # int(math.sqrt(fvectors_test.shape[0])) # 9 gives 57% for low quality

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
    Search for word in the grid of the previously classified letter labels. 

    Args:
        labels (np.ndarray) : 2D array that stores a classified letter in each 
                              square of the wordsearch puzzle.
        words (list[str]) : List of words to be found in the word-search puzzle.
        model (dict) : The model parameters learned during training.

    Returns:
        words_pos (list[tuple]): List of four-element tuples indicating each 
                                 word's position in the puzzle grid.
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
    of a word in the puzzle grid (label grid) by matching the length of the 
    targetted word. Then, each match is ranked by their correctness (i.e. how many
    letters are the same as in the searched word) and the highest scoring match 
    is returned.

    Args:
        word (str) : target word to be found in the label grid.
        label_grid (np.ndarray) : the grid of classified labels that construct the
                                  puzzle grid for the word search game.

    Returns:
        cm_pos (tuple) : the position in the label grid of a string that is the closest
                         match for the target word; the position is stored in a tuple
                         in the format (r1, c1, r2, c2), where r1 and c1 determine the
                         position (in row and column) of the first letter, and r2 and c2
                         determine th position of the ending letter of the word.
    """
    wlen = len(word)
    nrow, ncol = label_grid.shape
    # Compute all possible starting and ending indices of any string in a label
    # grid for both row and column of the label grid using Cartesian Product.
    irows = [r for r in product(list(range(nrow)), repeat=2)]
    icols = [c for c in product(list(range(ncol)), repeat=2)]

    # Combine above to find all possible staring and ending positions of the 
    # searched word in the grid considering all directions (diagonals, rows,
    # columns) in which the word might be placed and matching the word length.
    expected_word_pos = [
        (r1, c1, r2, c2) for c1, c2 in icols for r1, r2 in irows
            if (abs(r1 - r2) == (wlen - 1) and abs(c1 - c2) == (wlen - 1))    # diag
                or (abs(r1 - r2) == (wlen - 1) and c1 == c2)                  # col
                or (abs(c1 - c2) == (wlen - 1) and r1 == r2)                  # row
    ]

    # Create all possible word guesses from subsequent elements in the grid. Match
    # the target word length and track the positions of each computed guess in dict.
    wmatch_pos_dict = {}
    for pos in expected_word_pos:
        r1, c1, r2, c2 = pos

        # Set up list of row and column coordinates of each letter of a word guess.
        # And construct a word guess accordingly.
        irow = setup_coords(r1, r2, wlen)
        icol = setup_coords(c1, c2, wlen)
        wguess = "".join([label_grid[r, c] for r, c in zip(irow, icol)])
        wmatch_pos_dict[wguess] = pos

    # Find closest match among all, then return its coordinates by matching the key.
    closest_match = find_closest_match(word, wmatch_pos_dict.keys())
    cm_pos = wmatch_pos_dict[closest_match]
    return cm_pos


def setup_coords(coor1, coor2, wlen) -> List[int]:
    """
    Set up list of start to end coordinates in a row or column of the label grid.

    Args:
        coor1 (int) :
        coor2 (int) :
        wlen (int) : Length of the searched word.

    Returns:
        icoords (List[int]) : 
    """
    icoords = []
    if coor1 < coor2:
        icoords = list(range(coor1, (coor2 + 1)))
    elif coor1 > coor2:
        icoords = list(range(coor1, (coor2 - 1), -1))
    else:
        icoords = [coor1] * wlen
    return icoords


def find_closest_match(searched_word: str, word_guesses: List[str]) -> str:
    """
    Find the closest match for a target word by comparing how many letters in each
    word guess from the list are the same as in the searched word. The guess with 
    the highest scoring comparison is returned. In case of more than one guesses
    having the same max score, a random one out of the highest scoring is returned.

    Args:
        searched_word (str) : Target word to be matched.
        word_guesses List[str] : List of guessed words based on sequences of letters
                                 in the label grid in all directions; each guess 
                                 match the length of the target.

    Returns:
        closest_match : closest match to the searched word found by the max score.
    """
    wguess_score_dict = {}

    # For each guess, calculate how many letters in it for each position match 
    # the ones of the target word.
    for wg in word_guesses:
        wg_score = sum(
            wg[i] == searched_word[i] for i in range(len(wg))
        )
        wguess_score_dict[wg] = wg_score

    # Find maximum score value. Select all keys that have max_score as their value.
    max_wg_score = max(wguess_score_dict.values())
    closest_matches = [wg for wg, s in wguess_score_dict.items() if s == max_wg_score]

    # If there are more than 1 keys with the max value. Randomize the procedure and select
    # a random key with the max score. Otherwise, return the single key matching the max.
    if len(closest_matches) > 1:
        closest_match = np.random.choice(closest_matches)
        return closest_match
    return closest_matches[0]
