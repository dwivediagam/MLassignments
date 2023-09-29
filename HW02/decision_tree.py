import numpy as np
from collections import Counter

class DecisionTreeNode:
    def __init__(self):
        self.is_leaf = True
        self.label = None
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    # print(counts)
    probs = counts / len(labels)
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain_ratio(data, labels, threshold):
    total_entropy = entropy(labels)

    # Split data and labels based on the threshold
    left_indices = data >= threshold
    right_indices = ~left_indices
    print(left_indices)
    print("Left Entropy: ",entropy(labels[np.where(left_indices)[0]]))
    print("Right Entropy: ",entropy(labels[np.where(right_indices)[0]]))
    # print(labels[np.where(left_indices)[0]])
    # if np.all(left_indices) or np.all(right_indices):
    #     return 0.0, float('inf')

    left_labels = labels[left_indices]
    right_labels = labels[right_indices]
    print(left_labels)
    print(right_labels)
    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)

    # Calculate information gain
    info_gain = total_entropy - ((len(left_labels) / len(labels)) * left_entropy +
                                 (len(right_labels) / len(labels)) * right_entropy)

    # Calculate split information
    split_info = -((len(left_labels) / len(labels)) * np.log2(len(left_labels) / len(labels) + 1e-10) +
                   (len(right_labels) / len(labels)) * np.log2(len(right_labels) / len(labels) + 1e-10))

    # Calculate information gain ratio
    gain_ratio = info_gain / (split_info + 1e-10)

    return info_gain, gain_ratio

def find_best_split(data, labels):
    num_samples, num_features = data.shape
    best_gain_ratio = 0.0
    best_threshold = None
    best_feature = None

    for feature in range(num_features):
        unique_values = np.unique(data[:, feature])
        for threshold in unique_values:
            gain, gain_ratio = information_gain_ratio(data[:, feature], labels, threshold)

            if gain_ratio > best_gain_ratio:
                best_gain_ratio = gain_ratio
                best_threshold = threshold
                best_feature = feature

    return best_feature, best_threshold

def build_decision_tree(data, labels):
    node = DecisionTreeNode()

    if len(np.unique(labels)) == 1:
        node.label = labels[0]
        return node

    if len(data) == 0:
        node.label = 1  # Predict class 1 if no majority class in the leaf
        return node

    best_feature, best_threshold = find_best_split(data, labels)

    if best_feature is None or best_threshold is None:
        node.label = 1  # Predict class 1 if no valid split found
        return node

    node.is_leaf = False
    node.split_feature = best_feature
    node.split_threshold = best_threshold

    left_indices = data[:, best_feature] < best_threshold
    right_indices = ~left_indices

    node.left = build_decision_tree(data[left_indices], labels[left_indices])
    node.right = build_decision_tree(data[right_indices], labels[right_indices])

    return node

def predict(node, x):
    if node.is_leaf:
        return node.label

    if x[node.split_feature] < node.split_threshold:
        return predict(node.left, x)
    else:
        return predict(node.right, x)


data = []
labels = []
with open('Homework 2 data/Druns.txt', 'r') as file:
    for line in file:
        x1, x2, label = map(float, line.strip().split())
        data.append([x1, x2])
        labels.append(label)

data = np.array(data)
labels = np.array(labels)
print(data[0], labels[0])
print(entropy(labels))
# For data with features of first column
print("Information Gain Ratio: ", information_gain_ratio(np.array([x[1] for x in data]), labels, 8))
# For data with features of second column
print("Information Gain Ratio: ", information_gain_ratio(np.array([x[1] for x in data]), labels, 8))
# Build the decision tree
# root = build_decision_tree(data, labels)

# Example prediction
# example_input = [0.76, 0.9]
# prediction = predict(root, example_input)
# print("Prediction for input {}: {}".format(example_input, prediction))
