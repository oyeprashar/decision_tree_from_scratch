"""
Implement Decision Tree
"""

import numpy as np
from collections import Counter

class Node:
    def __init__(self):
        self.feature_index = None
        self.split_threshold = None
        self.label = None
        self.left = None
        self.right = None


class DecisionTree:

    def __init__(self, max_depth=100, min_samples_split=2, candidate_features_per_split=None):
        self.max_depth = max_depth
        self.candidate_features_per_split = candidate_features_per_split
        self.min_samples_split = min_samples_split
        self.root = None  # saved when we train and generate the actual tree

    def fit(self, X_train, y_train):

        # all the features are considered to figure out the best feature
        if self.candidate_features_per_split is None:
            self.candidate_features_per_split = X_train.shape[1]
        else:
            # we need to make sure candidate_features_per_split is not greater than number of features as we will be
            # using np.random.choice which breaks when we ask it to pick 10 things from a list of 5!
            self.candidate_features_per_split = min(self.candidate_features_per_split, X_train.shape[1])

        self.root = self.generate_tree(X_train, y_train)

    def predict(self, X):
        predictions = []
        for example in X:
            predictions.append(self.traverse(example, self.root))
        return predictions

    def traverse(self, X, root):

        if root.label is not None:
            return root.label

        feature_index = root.feature_index
        threshold = root.split_threshold

        # Since we are processing example by example, this is 1D array
        # and we can access the column of 1D array like this
        if X[feature_index] <= threshold:
            return self.traverse(X, root.left)
        else:
            return self.traverse(X, root.right)

    def most_common_label(self, y):
        return Counter(y).most_common()[0][0]  # (label, count)

    def compute_entropy(self, y):

        """
        entropy = -(p+)log(p+) - (p-)log(p-) <--- here p is proportions

        for more than 2 classes the formula becomes :
            entropy = - summation ( p_class * log(p_class))
        """

        freq_per_class = np.bincount(y)
        proportions = freq_per_class / len(y)

        # if p > 0 because log(0) will be inf
        return - np.sum([p * np.log(p) for p in proportions if p > 0])

    def information_gain(self, feature_col_data_all_rows, y_train, threshold):

        left_y_train_indices, right_y_train_indices = self.split_based_on_predicate(feature_col_data_all_rows,threshold)

        # to avoid division by zero error
        if len(left_y_train_indices) == 0 or len(right_y_train_indices) == 0:
            return 0

        entropy_parent = self.compute_entropy(y_train)
        left_entropy = self.compute_entropy(y_train[left_y_train_indices])  # the labels corresponding to rows of X are at col of y
        right_entropy = self.compute_entropy(y_train[right_y_train_indices])

        # information gain = parent entropy - weighted entropy of children
        children_entropy = (len(left_y_train_indices) / len(y_train)) * left_entropy + (
                    len(right_y_train_indices) / len(y_train)) * right_entropy
        return entropy_parent - children_entropy

    def get_best_feature_and_threshold_for_split(self, X_train, y_train, candidate_features_indices):

        """
            * We will process all the candidate features
            * There possible thresholds is set of unique values the feature takes across the dataset
            * We save the value and feature that gives the best information gain
        """

        best_gain = -1
        best_split_feature_index = None
        best_split_feature_threshold = None

        for col_index in candidate_features_indices:

            # all rows for this col_index
            feature_col_data_all_rows = X_train[:, col_index]
            possible_thresholds = np.unique(feature_col_data_all_rows)

            for threshold in possible_thresholds:
                # step 1 : compute the information gain
                information_gain = self.information_gain(feature_col_data_all_rows, y_train, threshold)

                # step 2 : save it if its better
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_split_feature_index = col_index
                    best_split_feature_threshold = threshold

        return best_split_feature_index, best_split_feature_threshold, best_gain

    def split_based_on_predicate(self, x, threshold):
        """
        x : is 1D array since X_train[:, best_feature_index] == 1D array
        It contains value of that best split feature for all the rows

        threshold : left is all the rows <= threshold and rest is right
        """

        left_row_indices = np.argwhere(x <= threshold).flatten()
        right_row_indices = np.argwhere(x > threshold).flatten()

        return left_row_indices, right_row_indices

    def generate_tree(self, X_train, y_train, depth=0):

        number_of_examples, number_of_features = X_train.shape[0], X_train.shape[1]

        # base case : return a leaf node
        if len(np.unique(y_train)) == 1 or depth >= self.max_depth or number_of_examples < self.min_samples_split:
            leaf_node = Node()
            leaf_node.label = self.most_common_label(y_train)
            return leaf_node

        # step 1 : pick the candidate features
        candidate_features_indices = np.random.choice(number_of_features, self.candidate_features_per_split, replace=False)

        # step 2 : out of these candidate features compute the best feature and its threshold <-- based on information gain
        best_feature_index, best_threshold, best_gain = self.get_best_feature_and_threshold_for_split(X_train, y_train, candidate_features_indices)

        if best_gain <= 0:
            leaf_node = Node()
            leaf_node.label = self.most_common_label(y_train)
            return leaf_node

        # step 3 : The current node will hold this feature's index and threshold
        currNode = Node()
        currNode.feature_index = best_feature_index
        currNode.split_threshold = best_threshold

        # step 4 : split the data based on the predicate and left and right of this new node will be generated recursively

        left_row_indices, right_row_indices = self.split_based_on_predicate(X_train[:, best_feature_index], best_threshold)

        # [left_row_indices, :] <-- select these rows and all the columns
        currNode.left = self.generate_tree(X_train[left_row_indices, :], y_train[left_row_indices], depth + 1)
        currNode.right = self.generate_tree(X_train[right_row_indices, :], y_train[right_row_indices], depth + 1)

        # step 5 : return current node
        return currNode
