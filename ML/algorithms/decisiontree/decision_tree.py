"""
Author: Philip Andreadis
e-mail: philip_andreadis@hotmail.com


Implementation of Decision Tree model from scratch.
Metric used to apply the split on the data is the Gini index which is calculated for each feature's single value
in order to find the best split on each step. This means there is room for improvement performance wise as this
process is O(n^2) and can be reduced to linear complexity.

Parameters of the model:
max_depth (int): Maximum depth of the decision tree
min_node_size (int): Minimum number of instances a node can have. If this threshold is exceeded the node is terminated

Both are up to the user to set.

Input dataset to train() function must be a numpy array containing both feature and label values.

"""


from collections import Counter
import numpy as np


class DecisionTree:
    def __init__(self, max_depth, min_node_size):
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.final_tree = {}

    """
        This function calculates the gini index of a split in the dataset
        Firstly the gini score is calculated for each child note and the resulting Gini is the weighted sum of gini_left and gini_right

        Parameters:
        child_nodes (list of np arrays): The two groups of instances resulting from the split

        Returns:
        float:Gini index of the split 

       """

    def calculate_gini(self, child_nodes):
        n = 0
        # Calculate number of all instances of the parent node
        for node in child_nodes:
            n = n + len(node)
        gini = 0
        # Calculate gini index for each child node
        for node in child_nodes:
            m = len(node)

            # Avoid division by zero if a child node is empty
            if m == 0:
                continue

            # Create a list with each instance's class value
            y = []
            for row in node:
                y.append(row[-1])

            # Count the frequency for each class value
            freq = Counter(y).values()
            node_gini = 1
            for i in freq:
                node_gini = node_gini - (i / m) ** 2
            gini = gini + (m / n) * node_gini
        return gini

    """
            This function splits the dataset on certain value of a feature
            Parameters:
            feature_index (int): Index of selected feature
            
            threshold : Value of the feature split point
            

            Returns:
            np.array: Two new groups of split instances

           """

    def apply_split(self, feature_index, threshold, data):
        instances = data.tolist()
        left_child = []
        right_child = []
        for row in instances:
            if row[feature_index] < threshold:
                left_child.append(row)
            else:
                right_child.append(row)
        left_child = np.array(left_child)
        right_child = np.array(right_child)
        return left_child, right_child

    """
                This function finds the best split on the dataset on each iteration of the algorithm by evaluating
                all possible splits and applying the one with the minimum Gini index.
                Parameters:
                data: Dataset

                Returns node (dict): Dictionary with the index of the splitting feature and its value and the two child nodes

               """

    def find_best_split(self, data):
        num_of_features = len(data[0]) - 1
        gini_score = 1000
        f_index = 0
        f_value = 0
        # Iterate through each feature and find minimum gini score
        for column in range(num_of_features):
            for row in data:
                value = row[column]
                l, r = self.apply_split(column, value, data)
                children = [l, r]
                score = self.calculate_gini(children)
                # print("Candidate split feature X{} < {} with Gini score {}".format(column,value,score))
                if score < gini_score:
                    gini_score = score
                    f_index = column
                    f_value = value
                    child_nodes = children
        # print("Chosen feature is {} and its value is {} with gini index {}".format(f_index,f_value,gini_score))
        node = {"feature": f_index, "value": f_value, "children": child_nodes}
        return node

    """
        This function calculates the most frequent class value in a group of instances
        Parameters:
        node: Group of instances

        Returns : Most common class value

    """

    def calc_class(self, node):
        # Create a list with each instance's class value
        y = []
        for row in node:
            y.append(row[-1])
        # Find most common class value
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]

    """
        Recursive function that builds the decision tree by applying split on every child node until they become terminal.
        Cases to terminate a node is: i.max depth of tree is reached ii.minimum size of node is not met iii.child node is empty
        Parameters:
        node: Group of instances
        depth (int): Current depth of the tree


    """

    def recursive_split(self, node, depth):
        l, r = node["children"]
        del node["children"]
        if l.size == 0:
            c_value = self.calc_class(r)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        elif r.size == 0:
            c_value = self.calc_class(l)
            node["left"] = node["right"] = {"class_value": c_value, "depth": depth}
            return
        # Check if tree has reached max depth
        if depth >= self.max_depth:
            # Terminate left child node
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
            # Terminate right child node
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
            return
        # process left child
        if len(l) <= self.min_node_size:
            c_value = self.calc_class(l)
            node["left"] = {"class_value": c_value, "depth": depth}
        else:
            node["left"] = self.find_best_split(l)
            self.recursive_split(node["left"], depth + 1)
        # process right child
        if len(r) <= self.min_node_size:
            c_value = self.calc_class(r)
            node["right"] = {"class_value": c_value, "depth": depth}
        else:
            node["right"] = self.find_best_split(r)
            self.recursive_split(node["right"], depth + 1)

    """
        Apply the recursive split algorithm on the data in order to build the decision tree
        Parameters:
        X (np.array): Training data
        
        Returns tree (dict): The decision tree in the form of a dictionary.
    """

    def train(self, X):
        # Create initial node
        tree = self.find_best_split(X)
        # Generate the rest of the tree via recursion
        self.recursive_split(tree, 1)
        self.final_tree = tree
        return tree

    """
        Prints out the decision tree.
        Parameters:
        tree (dict): Decision tree

    """

    def print_dt(self, tree, depth=0):
        if "feature" in tree:
            print(
                "\nSPLIT NODE: feature #{} < {} depth:{}\n".format(
                    tree["feature"], tree["value"], depth
                )
            )
            self.print_dt(tree["left"], depth + 1)
            self.print_dt(tree["right"], depth + 1)
        else:
            print(
                "TERMINAL NODE: class value:{} depth:{}".format(
                    tree["class_value"], tree["depth"]
                )
            )

    """
        This function outputs the class value of the instance given based on the decision tree created previously.
        Parameters:
        tree (dict): Decision tree
        instance(id np.array): Single instance of data

        Returns (float): predicted class value of the given instance
    """

    def predict_single(self, tree, instance):
        if not tree:
            print("ERROR: Please train the decision tree first")
            return -1
        if "feature" in tree:
            if instance[tree["feature"]] < tree["value"]:
                return self.predict_single(tree["left"], instance)
            else:
                return self.predict_single(tree["right"], instance)
        else:
            return tree["class_value"]

    """
        This function outputs the class value for each instance of the given dataset.
        Parameters:
        X (np.array): Dataset with labels
        
        Returns y (np.array): array with the predicted class values of the dataset
    """

    def predict(self, X):
        y_predict = []
        for row in X:
            y_predict.append(self.predict_single(self.final_tree, row))
        return np.array(y_predict)


if __name__ == "__main__":

    # # test dataset
    # X = np.array([[1, 1,0], [3, 1, 0], [1, 4, 0], [2, 4, 1], [3, 3, 1], [5, 1, 1]])
    # y = np.array([0, 0, 0, 1, 1, 1])

    train_data = np.loadtxt("example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("example_data/targets.txt")

    # Build tree
    dt = DecisionTree(5, 1)
    tree = dt.train(train_data)
    y_pred = dt.predict(train_data)
    print(f"Accuracy: {sum(y_pred == train_y) / train_y.shape[0]}")
    # Print out the decision tree
    # dt.print_dt(tree)
