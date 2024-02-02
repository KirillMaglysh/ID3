from random import randint

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

from DecisionTree import DecisionTree


def select_n_random_items(x, y, n) -> ([], []):
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


def train_and_test(training_data, training_categories, test_data, test_categories) -> DecisionTree:
    clf = tree.DecisionTreeClassifier().fit(training_data, training_categories)
    my_tree = DecisionTree(training_data, training_categories)

    my_ans = []
    for item in test_data:
        my_ans.append(my_tree.predict(item))

    print(f"sklearn accuracy: {calc_accuracy(my_ans, test_categories)}")
    print(f"Kirill accuracy: {calc_accuracy(clf.predict(test_data), test_categories)}")
    return my_tree


def solve_full():
    iris = load_iris()
    training_data, training_categories = iris.data, iris.target
    test_data, test_categories = select_n_random_items(training_data, training_categories, 40)
    print("\n################## Testing on full-parameter data ##################")
    train_and_test(training_data, training_categories, test_data, test_categories)
    print("################## Testing on full-parameter data ##################\n")


def choose_dimensions_for_array(arr: [], dimensions: []) -> []:
    result = []
    for dim in dimensions:
        result.append(arr[dim])

    return result


def choose_dimensions_for_table(data: [[]], dimensions: []) -> [[]]:
    result = []
    for arr in data:
        result.append(choose_dimensions_for_array(arr, dimensions))

    return result


def draw_plot(my_tree: DecisionTree, test_data: [[]]):
    fig, axis = plt.subplots()
    my_tree.draw_plot(axis, 4.2, 3)

    for item in test_data:
        category = my_tree.predict(item)
        if category == 0:
            col = 'red'
        elif category == 1:
            col = 'blue'
        else:
            col = 'green'

        plt.scatter(item[0], item[1], c=col)

    fig.show()


def solve2d():
    iris = load_iris()
    training_categories = iris.target
    training_data = choose_dimensions_for_table(iris.data, [1, 3])

    test_data, test_categories = select_n_random_items(training_data, training_categories, 40)

    print("\n################## Testing on 2-parameter data ##################")
    my_tree = train_and_test(training_data, training_categories, test_data, test_categories)
    print("################## Testing on 2-parameter data ##################\n")

    draw_plot(my_tree, test_data)


def draw_tree():
    iris = load_iris()
    training_data, training_categories = iris.data, iris.target
    my_tree = DecisionTree(training_data, training_categories)
    my_tree.draw_tree()


solve_full()
solve2d()
# draw_tree()
