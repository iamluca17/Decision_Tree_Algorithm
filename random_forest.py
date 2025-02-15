import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score, mean_squared_error, root_mean_squared_error

from sklearn.model_selection import train_test_split

from collections import defaultdict

class Random_Forest():

    def __init__(self, tree_min_samples=2, tree_max_depth=2, mode="classify", tree_thresh_quantile_opt=False, n_trees=10, max_samples = None, max_features = None, random_state=None, weigh_predictions=False):

        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.weigh_predictions = weigh_predictions

        self.tree_settings = {
        "min_samples": tree_min_samples,
        "max_depth": tree_max_depth,
        "mode": None,
        "tree_thresh_quantile_opt": tree_thresh_quantile_opt
        }

        if mode == "classify" or mode == "regression":
            self.tree_settings["mode"] = mode
        else:
            raise Exception("Invalid mode! Model supports classification or regression functionality only!")

        self.trees = []
        self.trees_with_weights = []

    
    def fit(self, features, outcomes, Decision_Tree_class):

        if self.weigh_predictions:
            self.fit_w_validation(features, outcomes, Decision_Tree_class)
        else:
            self.fit_wo_validation(features, outcomes, Decision_Tree_class)
    
    def predict_from_dataset(self, X):

        if self.weigh_predictions:
            return self.weighted_pred_aggregation(X)
        else:
            return self.unweighted_pred_aggregation(X)

    def fit_wo_validation(self, features, outcomes, Decision_Tree_class):

        np.random.seed(self.random_state)

        n_samples, n_features = features.shape
        self.trees = []

        if self.max_samples is None:
            self.max_samples = n_samples

        if self.max_features is None:
            if self.tree_settings["mode"] == "classify":
                self.max_features = max(1, int(np.sqrt(n_features)))
            elif self.tree_settings["mode"] == "regression":
                self.max_features = max(1, int(n_features / 3))

        for _ in range(self.n_trees):


            selected_features_i = np.random.choice(n_features, self.max_features, replace=False)
            selected_samples_i = np.random.choice(n_samples, self.max_samples, replace=True)

            features_sample, outcomes_sample = features[selected_samples_i][:, selected_features_i], outcomes[selected_samples_i]

            tree = Decision_Tree_class(self.tree_settings["min_samples"], self.tree_settings["max_depth"], 
                                       self.tree_settings["mode"], self.tree_settings["tree_thresh_quantile_opt"])

            tree.fit(features_sample, outcomes_sample)

            self.trees.append((tree, selected_features_i))

    
    
    def unweighted_pred_aggregation(self, X):
    
        parallel_predictions = np.array([tree.predict_from_dataset(X[:, selected_features]) for tree, selected_features in self.trees])

        if self.tree_settings["mode"] == "classify":
            from scipy.stats import mode
            return mode(parallel_predictions, axis=0)[0].flatten()
        elif self.tree_settings["mode"] == "regression":
            return np.mean(parallel_predictions, axis=0)

    def fit_w_validation(self, features, outcomes, Decision_Tree_class):

        np.random.seed(self.random_state)

        train_f, val_f, train_o, val_o = train_test_split(features, outcomes, test_size=.2 ,random_state=42)

        n_samples, n_features = train_f.shape
        self.trees_with_weights = []
        val_accuracies = []

        
        if self.max_samples is None:
            self.max_samples = n_samples

        if self.max_samples is not None:
            while self.max_samples > n_samples:
                self.max_samples = self.max_samples*0.8

        if self.max_features is None:
            if self.tree_settings["mode"] == "classify":
                self.max_features = max(1, int(np.sqrt(n_features)))
            elif self.tree_settings["mode"] == "regression":
                self.max_features = max(1, int(n_features / 3))

        
        for _ in range(self.n_trees):


            selected_features_i = np.random.choice(n_features, self.max_features, replace=False)
    
            # Ensure that the number of samples selected doesn't exceed available samples
            selected_samples_i = np.random.choice(n_samples, self.max_samples, replace=True)
            

            if len(selected_samples_i) > len(train_o):
                raise ValueError("selected_samples_i exceeds the bounds of the outcomes array")
        
            # Sample the selected rows and features
            features_sample = train_f[selected_samples_i][:, selected_features_i]
            outcomes_sample = train_o[selected_samples_i]

            tree = Decision_Tree_class(self.tree_settings["min_samples"], self.tree_settings["max_depth"], 
                                       self.tree_settings["mode"], self.tree_settings["tree_thresh_quantile_opt"])

            tree.fit(features_sample, outcomes_sample)

            val_predictions = tree.predict_from_dataset(val_f)

            weight = 0
            if self.tree_settings["mode"] == "classify":
                accuracy = accuracy_score(val_o, val_predictions)
                weight = accuracy  # Higher accuracy → Higher weight
            elif self.tree_settings["mode"] == "regression":  # Regression mode
                rmse = mean_squared_error(val_o, val_predictions)
                weight = 1 / (rmse + 1e-10)

            val_accuracies.append(weight)
            
            #self.trees_with_weights is a pair of a pair. This pairs a pair of a tree and it's selected features with a weight
            #initially that weight will be just the accuracy and will be divided by the sum of accuracies after for loop
            self.trees_with_weights.append((tree, selected_features_i, weight))

        val_accuracies = np.array(val_accuracies)

        summed_accuracy = np.sum(val_accuracies)
        if summed_accuracy == 0:
            summed_accuracy = 1

        self.trees_with_weights = [(tree, tree_features, accuracy/summed_accuracy) for tree, tree_features, accuracy in self.trees_with_weights]
    


    

    def weighted_pred_aggregation(self, X):
    
        

        n_samples = X.shape[0]
        class_votes = [defaultdict(float) for _ in range(n_samples)]

        for tree, selected_features, weight in self.trees_with_weights:
            pred = np.array(tree.predict_from_dataset(X[:, selected_features]))
            weight = np.float64(weight)

            for i, prediction in enumerate(pred):
                class_votes[i][prediction] += weight

        if self.tree_settings["mode"] == "classify":

            final_predictions = np.array([max(votes, key=votes.get) for votes in class_votes])
        elif self.tree_settings["mode"] == "regression":
            final_predictions = np.array([
                sum(value * weight for value, weight in votes.items()) / sum(votes.values())
                for votes in class_votes
            ])

        return final_predictions
        
        
