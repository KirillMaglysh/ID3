from sklearn.datasets import load_iris

from DecisionTree import DecisionTree

iris = load_iris()
X, y = iris.data, iris.target

tre = DecisionTree(X, y)

print('a')