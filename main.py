from random import randint

import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

from DecisionTree import DecisionTree


def select_n_random_items(x, y, n):
    xTest = []
    yTest = []
    for i in range(0, n):
        idx = randint(0, len(x) - 1)
        xTest.append(x[idx])
        yTest.append(y[idx])
        np.delete(x, idx)
        np.delete(y, idx)
    return xTest, yTest


def calc_accuracy(ideal: [], real: []) -> float:
    correct_number = 0
    for i in range(0, len(ideal)):
        correct_number += ideal[i] == real[i]
    return correct_number / len(ideal)


def solve_full():
    iris = load_iris()
    training_data, training_categories = iris.data, iris.target,

    clf = tree.DecisionTreeClassifier().fit(training_data, training_categories)
    my_tree = DecisionTree(training_data, training_categories)

    test_data, test_categories = select_n_random_items(training_data, training_categories, 40)
    my_ans = []
    for item in test_data:
        my_ans.append(my_tree.predict(item))

    print("\n################## Testing on full-parameter data ##################")
    print(f"sklearn accuracy: {calc_accuracy(my_ans, my_ans)}")
    print(f"Kirill accuracy: {calc_accuracy(clf.predict(test_data), my_ans)}")
    print("################## Testing on full-parameter data ##################\n")


solve_full()
