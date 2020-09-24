"""
Author: Philip Andreadis
e-mail: philip_andreadis@hotmail.com


Implementation of Random Forest model from scratch.
The DecisionTree class from this project is used for generating the trees of the random forest.
This class remains with no changes as the dataset is split into a number of folds with a random subset of features on which each tree is trained on.
As a result each tree is trained on a different group of the dataset in order to avoid correlation between them.
The predicted class value of each instance is chosen by voting from each single tree's outcome.

Parameters of the model:
MAX_DEPTH (int): Maximum depth of the decision tree
MIN_NODE (int): Minimum number of instances a node can have. If this threshold is exceeded the node is terminated
FOLD_SIZE (int): Value between 1-10 representing the percentage of the original dataset size each fold should be.
N_TREES (int):The toral number of trees that will be trained.

Input dataset to train() function must be a numpy array containing both feature and label values.

"""



from random import randrange
from random import randint
import numpy as np
from decision_tree import DecisionTree

# fold size (% of dataset size) e.g. 3 means 30%
FOLD_SIZE = 10
# number of trees
N_TREES = 20
# max tree depth
MAX_DEPTH = 30
# min size of tree node
MIN_NODE = 1


class RandomForest:
    def __init__(self,n_trees,fold_size):
        self.n_trees = n_trees
        self.fold_size = fold_size
        self.trees = list()



    """
        This function splits the given dataset into n-folds with replacement. The number of folds is equal to the number of the trees that will be trained.
        Each tree will have one fold as input. The size of the folds is a percentage (p) of the size of the original dataset. 

        Parameters:
        dataset: np array of the given dataset
        n_folds (int): number of folds in which the dataset should be split. Must be equal to the number of trees the user wants to train
        p (int): suggests the percentage of the dataset's size the size of a single fold should be.

        Returns list of np arrays: list with the k-folds 

    """
    def cross_validation_split(self,dataset, n_folds, p):
        dataset_split = list()
        fold_size = int(len(dataset)*p/10)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset))
                fold.append(dataset[index])
            set = np.array(fold)
            dataset_split.append(set)
        return dataset_split


    """
        This function randomizes the selection of the features each tree will be trained on.

        Parameters:
            splits list of np arrays: list of folds
            

        Returns list of np arrays: list with the k-folds with some features randomly removed

    """
    def randomize_features(self,splits):
        dataset_split = list()
        l = len(splits[0][0])
        n_features = int((l-1)*5/10)
        for split in splits:
            for i in range(n_features):
                rng = list(range(len(split[0]) - 1))
                selected = rng.pop(randint(0,len(rng)-1))
                split = np.delete(split, selected, 1)
            set = np.array(split)
            dataset_split.append(set)
        return dataset_split


    """
        Prints out all the decision trees of the random forest.
            
        BUG: The feature number is not representative of its initial enumeration in the original dataset due to the randomization. 
             This means that we do not know on which features each tree is trained on.
    """
    def print_trees(self):
        i = 1
        for t in self.trees:
            print("Tree#",i)
            temp = t.final_tree
            t.print_dt(temp)
            print("\n")
            i = i+1

    """
        Iteratively train each decision tree.
        Parameters:
        X (np.array): Training data

    """
    def train(self,X):
        train_x = self.cross_validation_split(X,self.n_trees,self.fold_size)
        train_x = self.randomize_features(train_x)
        for fold in train_x:
            dt = DecisionTree(MAX_DEPTH, MIN_NODE)
            dt.train(fold)
            self.trees.append(dt)


    """
        This function outputs the class value for each instance of the given dataset as predicted by the random forest algorithm.
        Parameters:
        X (np.array): Dataset with labels

        Returns y (np.array): array with the predicted class values of the dataset
    """
    def predict(self,X):
        predicts = list()
        final_predicts = list()
        for tree in self.trees:
            predicts.append(tree.predict(X))
        # iterate through each tree's class prediction and find the most frequent for each instance
        for i in range(len(predicts[0])):
            values = list()
            for j in range(len(predicts)):
                values.append(predicts[j][i])
            final_predicts.append(max(set(values), key=values.count))
        return final_predicts,predicts



if __name__ == "__main__":


    # Training data
    train_data = np.loadtxt("example_data/data.txt", delimiter=",")
    train_y = np.loadtxt("example_data/targets.txt")

    mock_train = np.loadtxt("example_data/mock_data.csv", delimiter=",")
    mock_y = mock_train[ : , -1]

    # Build and train model
    rf = RandomForest(N_TREES,FOLD_SIZE)
    rf.train(mock_train)

    # Evaluate model on training data
    y_pred,y_pred_ind = rf.predict(mock_train)
    print(f"Accuracy of random forest: {sum(y_pred == mock_y) / mock_y.shape[0]}")
    print("\nAccuracy for each individual tree:")
    c = 1
    for i in y_pred_ind:
        print("\nTree",c)
        print(f"Accuracy: {sum(i == mock_y) / mock_y.shape[0]}")
        c = c+1
