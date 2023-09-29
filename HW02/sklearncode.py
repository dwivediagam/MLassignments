import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = []
# labels = []
with open('/Users/agamdwivedi/Desktop/760/Assignment2/Homework 2 data/Dbig.txt', 'r') as file:
    for line in file:
        x1, x2, label = map(float, line.strip().split())
        data.append([x1, x2, label])
#         labels.append(label)

# For Sample=2048
data = data[:2048]
data = np.array(data)
# print(data[2])
# labels = np.array(labels)
# Assuming the last column is the target variable
X = data[:, :-1]  # Features
y = data[:, -1]   # Labels
# print(y.shape)
# Create a decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(10, 8))
plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['0', '1'])
plt.title("Decision Tree Diagram")
plt.show()
