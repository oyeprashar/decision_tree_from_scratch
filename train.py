import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decision_tree import DecisionTree

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

dt = DecisionTree(max_depth=10)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test)
print("accuracy is :", acc)