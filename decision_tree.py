import numpy as np

import pandas as pd

class Node():

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):

        #these parameters are important if the node is decision node 
        self.feature_index = feature_index #each node compares splits for one set of features
        self.threshold = threshold #threshold defines splits and is adjusted to obtain best split - this is the actual learning part
        self.left = left
        self.right = right
        self.info_gain = info_gain #measured by entropy reduction i.e. how much the split reduced entropy is how much information has been gained

        #self.value is used if node is leaf node
        self.value = value



class Decision_Tree_Classifier():

    def __init__(self, min_samples=2, max_depth=2):

        self.root = None

        self.min_samples = min_samples

        #too big of a max_depth can cause overfitting
        self.max_depth = max_depth


    def build_tree(self, dataset:pd.DataFrame, curr_depth=0):

        #note dataset is a a pandas dataframe

        features = dataset[:, :-1]
        targets = dataset[:,-1]
        num_samples, num_features = np.shape(features)

        if num_samples >= self.min_samples and curr_depth<= self.max_depth:

            #generate best split
            best_split = self.gen_best_split(dataset, num_samples, num_features)

            '''
            if information gain value is less then 0 
            there is no information gain because the data is pure 
            i.e it's of the same class so no more need for splitting
            '''
            if best_split["info_gain"] > 0: 
                
                #recursion for left child node
                left_child = self.build_tree(best_split["left_child_dataset"], curr_depth+1)
                #recursion for right child node
                right_child = self.build_tree(best_split["right_child_dataset"], curr_depth+1)

                #return node in case node is decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_child, right_child, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(targets)

        #return node in case node is leaf node
        return Node(value=leaf_value)

    def calculate_leaf_value(self, leaf_dataset):
        
        #this calculates the leaf value
        leaf_dataset = list(leaf_dataset)
        return max(leaf_dataset, key=leaf_dataset.count)

    def gen_best_split(self, dataset, num_samples, num_features):
        
        best_split = {}
        #max_info_gain is first initialized at -infinity
        max_info_gain = -float("inf")
        
        '''
        Check all possible thresholds and feature combinations 
        by looping through all feature sets and all thresholds in each feature set
        '''

        for feature_index in range(num_features):

            feature_values = dataset[:, feature_index]
            #possible thresholds can have values only from the feature set at the current feature index
            possible_thresholds  = np.unique(feature_values)

            for threshold in possible_thresholds:

                #split according to threshold
                left_child_dataset, right_child_dataset = self.split(dataset, feature_index, threshold)

                if len(left_child_dataset) > 0 and len(right_child_dataset) > 0:

                    #get the outcome targets for each sample
                    parent_outcome_target = dataset[:, -1]
                    left_child_outcome_target = left_child_dataset[:, -1]
                    right_child_outcome_target = right_child_dataset[:, -1]

                    curr_split_info_gain = self.info_gain(parent_outcome_target, left_child_outcome_target, right_child_outcome_target, "entropy")

                    #compare split information gain
                    if curr_split_info_gain > max_info_gain:

                        #save best slit information
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_child_dataset"] = left_child_dataset
                        best_split["right_child_dataset"] = right_child_dataset
                        best_split["info_gain"] = curr_split_info_gain

                        max_info_gain = curr_split_info_gain


        return best_split


    def split(self, dataset, feature_index, threshold):

        sub_threshold_dataset = np.array([row for row in dataset if row[feature_index] <= threshold])
        supra_threshold_dataset = np.array([row for row in dataset if row[feature_index] > threshold])

        return sub_threshold_dataset, supra_threshold_dataset


    #compute information gain for from parent to children after split as entropy or gini index difference
    def info_gain(self, parent_dataset_outcomes, left_child_dataset_outcomes, right_child_dataset_outcomes, mode="entropy"):

        weight_left = len(left_child_dataset_outcomes) / len(parent_dataset_outcomes)
        weight_right = len(right_child_dataset_outcomes) / len(parent_dataset_outcomes)

        if mode=="entropy":
            information_gain = self.entropy(parent_dataset_outcomes) - (weight_left*self.entropy(left_child_dataset_outcomes) + weight_right*self.entropy(right_child_dataset_outcomes))
        elif mode=="gini":
            information_gain = self.gini_index(parent_dataset_outcomes) - (weight_left*self.gini_index(left_child_dataset_outcomes) + weight_right*self.gini_index(right_child_dataset_outcomes))
        
        return information_gain


    #compute entropy
    '''
    Entropy Formula:
     
    H(S) = - Σ p(c) * log2(p(c))
    
    Where:
        H(S) is the entropy of the set S,
        p(c) is the probability of class c in the set S,
        log2(p(c)) is the binary logarithm of the probability of class c.
        -log2(p(c)) calculates numbers of bits describing the probability of class c
    '''
    def entropy(self, dataset_outcomes):

        class_labels = np.unique(dataset_outcomes)

        entropy = 0
        for class_label in class_labels:
            p_class = len(dataset_outcomes[dataset_outcomes==class_label])/len(dataset_outcomes)
            entropy += -p_class * np.log2(p_class)

        return entropy

    '''
    gini index is computed faster than entropy which uses logarithms so it's an alternative for optimizing computation time
    '''
    def gini_index(self, dataset_outcomes):

        class_labels = np.unique(dataset_outcomes)
        gini = 0

        for class_label in class_labels:
            p_class = len(dataset_outcomes[dataset_outcomes==class_label])/len(dataset_outcomes)
            gini += p_class**2
        
        return 1 - gini

    #trains the model
    def fit(self, X, Y):

        dataset = np.concatenate((X,Y), axis = 1)
        self.root = self.build_tree(dataset)

    #generates outcome vector from feature matrix
    def predict_from_dataset(self, X):
        
        predictions = [self.make_predictions(x, self.root) for x in X]

        return predictions
    
    #generates an out come for a sample features vector
    def make_predictions(self, x, tree:Node):

        if tree.value is not None:
            return tree.value
        
        if x[tree.feature_index] > tree.threshold:
            return self.make_predictions(x, tree.right)
        elif x[tree.feature_index] <= tree.threshold:
            return self.make_predictions(x, tree.left)


    #prints conceptual representaion of decision nodes and leaf nodes
    def print_tree(self, tree:Node=None, depth=0):

        if not tree:
            tree = self.root

        indent = "   " * depth

        if tree.value is not None:
            print(f"{indent}Leaf: {tree.value}")
        else:
            print(f"{indent}Node at depth {depth}: if x{tree.feature_index} <= {tree.threshold}:")
            self.print_tree(tree.left, depth+1)
            print(f"{indent}Node at depth {depth}: else if x{tree.feature_index} > {tree.threshold}:")
            self.print_tree(tree.right, depth+1)
