import math

import graphviz


def get_uniq_items_with_number(data: []) -> dict:
    items = {}
    for x in data:
        if items.__contains__(x):
            items[x] = items[x] + 1
        else:
            items[x] = 1

    return items


def calculate_array_entropy(data: []) -> float:
    items = get_uniq_items_with_number(data)
    entropy = 0
    setSize = len(data)
    for key in items:
        proportion = items[key] / setSize
        entropy -= proportion * math.log2(proportion)

    return entropy


def most_common_item(data: []) -> int:
    if len(data) == 0:
        return -1

    items = get_uniq_items_with_number(data)
    cnt = 0
    x = data[0]
    for key in items:
        if cnt < items[key]:
            cnt = items[key]
            x = key

    return x


def split_data_by_border(data: [[]], categories: [], best_dim: int, split_border: int) -> ([[]], [], [[]], []):
    left_data = []
    left_categories = []
    right_data = []
    right_categories = []
    for i in range(0, len(data)):
        if data[i][best_dim] <= split_border:
            left_data.append(data[i])
            left_categories.append(categories[i])
        else:
            right_data.append(data[i])
            right_categories.append(categories[i])

    return left_data, left_categories, right_data, right_categories


def calc_gain(original: [], left_items: [], right_items: []) -> float:
    left_entropy = calculate_array_entropy(left_items)
    right_entropy = calculate_array_entropy(right_items)
    initial_entropy = calculate_array_entropy(original)
    items_number = len(original)

    return initial_entropy - (len(left_items) / items_number) * left_entropy - (
            len(right_items) / items_number) * right_entropy


def split_categories_by_border(data: [[]], categories: [], dim: int, border: float) -> ([int], [int]):
    left_items = []
    right_items = []
    for i in range(0, len(data)):
        if data[i][dim] <= border:
            left_items.append(categories[i])
        else:
            right_items.append(categories[i])

    return left_items, right_items


def calc_dimension_gain(data: [[]], categories: [], dim: int, border: float) -> float:
    left_items, right_items = split_categories_by_border(data, categories, dim, border)
    return calc_gain(categories, left_items, right_items)


def find_split_for_dim(data: [[]], categories: [], dim: int) -> (float, float):
    best_gain = 0
    best_split = 0

    for i in range(0, len(data)):
        gain = calc_dimension_gain(data, categories, dim, data[i][dim])
        if gain > best_gain:
            best_gain = gain
            best_split = data[i][dim]

    return best_gain, best_split


class DecisionTree:
    LEAF_ENTROPY: float = 0.1
    MAX_DEPTH: int = 6

    class Node:
        def __init__(self):
            self.isLeaf: bool = False
            self.category: int = -1
            self.paramId: int = -1
            self.border: int = -1
            self.left_kid = None
            self.right_kid = None

    def __init__(self, data: [[]], categories: []):
        self.root = self.Node()
        self.__build_tree(data, categories, self.root)

    def predict(self, data) -> int:
        return self.__predict(self.root, data)

    def draw_tree(self):
        plot = graphviz.Digraph(comment='Decision tree built by ID3 algorithm')
        self.__draw_tree(plot, self.root, 0)
        plot.render(directory='tree-visual')

    def draw_plot(self, axis, max_x, max_y):
        self.__draw_plot(self.root, axis, max_x, max_y)

    def __build_tree(self, data: [[]], categories: [], node: Node, depth=0):
        if len(data) == 0 or depth > self.MAX_DEPTH or calculate_array_entropy(categories) <= self.LEAF_ENTROPY:
            self.__fill_leaf(node, most_common_item(categories))
            return

        left_data, left_categories, right_data, right_categories = self.split_data_for_information_gain_maximum(
            data, categories, node)

        node.left_kid = self.Node()
        self.__build_tree(left_data, left_categories, node.left_kid, depth + 1)

        node.right_kid = self.Node()
        self.__build_tree(right_data, right_categories, node.right_kid, depth + 1)

    @staticmethod
    def split_data_for_information_gain_maximum(data: [[]], categories: [], node: Node) -> (
            [[float]], [int], [[float]], [int]):
        dim_number = len(data[0])
        best_dim = 0
        info_gain = 0
        split_pos = 0
        for dim in range(0, dim_number):
            (info_gain_tmp, split_pos_tmp) = find_split_for_dim(data, categories, dim)
            if info_gain_tmp > info_gain:
                best_dim = dim
                info_gain = info_gain_tmp
                split_pos = split_pos_tmp

        node.paramId = best_dim
        node.border = split_pos
        return split_data_by_border(data, categories, best_dim, split_pos)

    @staticmethod
    def __fill_leaf(node: Node, category: int):
        node.isLeaf = True
        node.category = category

    def __predict(self, node, data, ) -> int:
        if node.isLeaf:
            return node.category

        if data[node.paramId] <= node.border:
            return self.__predict(node.left_kid, data)
        else:
            return self.__predict(node.right_kid, data)

    def __draw_tree(self, graph, node: Node, node_id: int):
        sub_tree_size = 1
        if node.isLeaf:
            graph.node(str(node_id), f'LEAF. TYPE= {node.category}')
            return sub_tree_size
        else:
            graph.node(str(node_id), f'Param= {node.paramId}, \nBorder= {node.border}')

        left_id = node_id + 1
        sub_tree_size += self.__draw_tree(graph, node.left_kid, left_id)
        right_id = node_id + sub_tree_size
        sub_tree_size += self.__draw_tree(graph, node.right_kid, right_id)

        graph.edge(str(node_id), str(left_id))
        graph.edge(str(node_id), str(right_id))

        return sub_tree_size

    def __draw_plot(self, node: Node, axis, max_x, max_y):
        if node.isLeaf:
            return

        if node.paramId == 0:
            axis.vlines(node.border, 0, max_y)
        else:
            axis.hlines(node.border, 0, max_x)

        self.__draw_plot(node.left_kid, axis, max_x, max_y)
        self.__draw_plot(node.right_kid, axis, max_x, max_y)
