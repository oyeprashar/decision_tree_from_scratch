import numpy as np
from collections import Counter


class Node:

    # Node(label=5) works because we added the * before it
    # this helps in creating a leaf node without passing other arugment
    def __init__(self, left = None, right = None, threshold = None, feature = None, *, label = None):
        self.left = left
        self.right = right

        # label if this node is a leaf node
        self.label = label #

        # the predicate this node uses to direct to flow to left or rigt
        self.threshold = threshold

        # The feature we are using to split
        # ex the root was language and this node uses python to split
        self.feature = feature


    def is_leaf_node(self):
        return self.label is not None


def most_common_label(y_train):
    # Give me the top 1 most frequent item.
    return Counter(y_train).most_common(1)[0][0] # (label, count)


class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):

        # if after all the predicates from the root have been applied
        # the num of training examples reaches node n > min_samples_split then we split
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

        # The number of features you consider to make the predicates based on them
        # ex if features where [salary, dob, experience] and we say 2 then whole tree
        # is made from randomly choosing 2 features out of 3
        # all predicates will be based on these two choosen features. None == consider all features
        self.features_per_split = n_features
        self.root = None

    def fit(self, X_train, y_train):

        if self.features_per_split is None:
            self.features_per_split = X_train.shape[1]
        else:
            self.features_per_split = min(X_train.shape[1], self.features_per_split)

        self.root = self.generate_tree(X_train, y_train)

    def generate_tree(self, X_train, y_train, depth = 0):

        num_of_samples, num_of_features = X_train.shape
        num_of_unique_labels = len(np.unique(y_train))

        # num_of_unique_labels == 1 -> all the samples can have this label! Pure leaf
        if depth >= self.max_depth or num_of_unique_labels == 1 or num_of_samples < self.min_samples_split:
            # for the current path i have reached the depth
            # i need to add a node with label for this branch
            common_label = most_common_label(y_train)
            return Node(label=common_label)

        # choosing the features to split on for the current node
        # replace = False means we select unique feature randomly and same feature is not picked more than once
        chosen_features = np.random.choice(num_of_features, self.features_per_split, replace = False)

        # Which feature is best to split on based on the information gain?
        # What is the threshold/predicate we need to make this split possible
        best_feature_index, best_threshold = self.select_best_feature_to_split(X_train, y_train, chosen_features)


        currNode = Node()
        currNode.feature = best_feature_index
        currNode.threshold = best_threshold # This is the column index

        # training example is  divided into left (where the predicate over the best feature matches) and right side( other wise)
        left_features_indices, right_features_indices = self.split_features(X_train[:,best_feature_index], best_threshold)

        # since y_train is also np array, is left_features_indices is array, it will give all the corresponding labels
        currNode.left = self.generate_tree(X_train[left_features_indices, :], y_train[left_features_indices], depth + 1)
        currNode.right = self.generate_tree(X_train[right_features_indices, :], y_train[right_features_indices], depth + 1)
        return currNode

    def split_features(self, X_column, best_threshold):
        left_row_indices = np.argwhere(X_column <= best_threshold).flatten()
        right_row_indices = np.argwhere(X_column > best_threshold).flatten()
        return left_row_indices, right_row_indices

    def select_best_feature_to_split(self, X_train, y_train, chosen_features):

        # chosen_features are chosen at random
        # we need to return the best feature to split on and the threshold
        best_gain = -1
        split_index, split_threshold = None, None

        for feature_index in chosen_features:
            X_column = X_train[:, feature_index]
            threshold_candidates = np.unique(X_column)

            for candidate_threshold in threshold_candidates:
                gain = self.compute_information_gain(candidate_threshold, X_column, y_train)
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold  = candidate_threshold

        return split_index, split_threshold

    # compute_information_gain : how much uncertainty the split removes
    def compute_information_gain(self, candidate_threshold, X_column, y_train):

        # gain(split) = entropy(root) - weighted sum of entropy of the child

        parent_entropy = self.entropy(y_train)
        left_row_indices, right_row_indices = self.split_features(X_column, candidate_threshold)

        if len(left_row_indices) == 0 or len(right_row_indices) == 0:
            return 0

        left_entropy = self.entropy(y_train[left_row_indices])
        right_entropy = self.entropy(y_train[right_row_indices])
        total_length = len(y_train)
        left_length = len(left_row_indices)
        right_length = len(right_row_indices)
        child_entropy =  ((left_length/total_length) * left_entropy) + ((right_length/total_length) * right_entropy)
        return parent_entropy - child_entropy

    def entropy(self, y_train):

        # for exactly two classes :  - (p+)log2(p+) - (p-)log2(p-)
        # for more than two classes : entropy = -sum(p_i * log(p_i))
        freq_per_class = np.bincount(y_train)
        proportions = freq_per_class / len(y_train)

        # entropy = -sum(p_i * log(p_i))
        return  - np.sum([p * np.log(p) for p in proportions if p > 0])


    def predict(self, X):
        predictions = np.array([])
        for x in X:
            predictions = np.append(predictions, self.traverse(self.root, x))
        return predictions


    def traverse(self, root, x):

        if root.is_leaf_node():
            return root.label

        # x is a single example and root.feature is col index
        # we find what the value is at that one example's root.feature index
        # this will be a single value which we compare with the threshold at the current node
        if x[root.feature] <= root.threshold:
            return self.traverse(root.left, x)
        else:
            return self.traverse(root.right, x)

