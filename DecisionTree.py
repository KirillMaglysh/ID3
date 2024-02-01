import math


def get_uniq_items_with_number_for_column(data, column: int) -> dict:
    items = {}
    dim1Size: int = len(data)
    for i in range(0, dim1Size):
        x = data[i][column]
        if items.__contains__(x):
            items[x] = items[x] + 1
        else:
            items[x] = 1

    return items


def get_uniq_items_with_number(data) -> dict:
    items = {}
    for x in data:
        if items.__contains__(x):
            items[x] = items[x] + 1
        else:
            items[x] = 1

    return items


def calculate_entropy_for_array(data) -> float:
    items = get_uniq_items_with_number(data)
    entropy = 0
    setSize = len(data)
    for key in items:
        proportion = items[key] / setSize
        entropy -= proportion * math.log2(proportion)

    return entropy


def most_common_item(data) -> int:
    items = get_uniq_items_with_number(data)
    cnt = 0
    x = data[0]
    for key in items:
        if cnt < items[key]:
            cnt = items[key]
            x = key

    return x


def split_by_param(data, categories, best_dim, split_val):
    left_data = []
    left_categories = []
    right_data = []
    right_categories = []
    for i in range(0, len(data)):
        if data[i][best_dim] <= split_val:
            left_data.append(data[i])
            left_categories.append(categories[i])
        else:
            right_data.append(data[i])
            right_categories.append(categories[i])

    return left_data, left_categories, right_data, right_categories


def calc_gain(data, categories, border, dim):
    setLess = []
    setMore = []
    sz = len(data)
    for i in range(0, sz):
        if data[i][dim] <= border:
            setLess.append(categories[i])
        else:
            setMore.append(categories[i])

    lEntropy = calculate_entropy_for_array(setLess)
    rEntropy = calculate_entropy_for_array(setMore)
    lSz = len(setLess)
    rSz = len(setMore)
    return calculate_entropy_for_array(categories) - (lSz / sz) * lEntropy - (rSz / sz) * rEntropy


def find_split_for_column(data, categories, dim):
    best_gain = 0
    best_split = 0

    for i in range(0, len(data)):
        gain = calc_gain(data, categories, data[i][dim], dim)
        if gain > best_gain:
            best_gain = gain
            best_split = data[i][dim]

    return best_gain, best_split


class DecisionTree:
    LEAF_ENTROPY: float = 0.1

    class Node:
        def __init__(self):
            self.isLeaf: bool = True
            self.category: int = -1
            self.paramId: int = -1
            self.border: int = -1
            self.leftKid = None
            self.rightKid = None

    def __init__(self, data, categories):
        self.root = self.Node()
        self.build_tree(data, categories, self.root)

    def build_tree(self, data, categories, node: Node):
        if len(data) == 0 or calculate_entropy_for_array(categories) <= self.LEAF_ENTROPY:
            node.isLeaf = True
            node.category = most_common_item(categories)
            return

        dim_number = len(data[0])

        best_dim = 0
        info_gain = 0
        split_pos = 0
        for dim in range(0, dim_number):
            (info_gain_tmp, split_pos_tmp) = find_split_for_column(data, categories, dim)
            if info_gain_tmp > info_gain:
                best_dim = dim
                info_gain = info_gain_tmp
                split_pos = split_pos_tmp

        node.paramId = best_dim
        node.border = split_pos
        (left_data, left_categories, right_data, right_categories) = split_by_param(data, categories, best_dim,
                                                                                    split_pos)

        node.leftKid = self.Node()
        self.build_tree(left_data, left_categories, node.leftKid)

        node.rightKid = self.Node()
        self.build_tree(right_data, right_categories, node.rightKid)

        return
