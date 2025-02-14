import numpy as np

import pandas as pd

class Random_Forest():

    def __init__(self, tree_min_samples=2, tree_max_depth=2, mode="classify", tree_thresh_quantile_opt=False, n_trees=10, max_samples = None, max_features = None, random_state=None):

        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state

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

    
    def fit(self, features, outcomes, Decision_Tree_class):

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

    
    
    def predict_from_dataset(self, X):
    
        parallel_predictions = np.array([tree.predict_from_dataset(X[:, selected_features]) for tree, selected_features in self.trees])

        if self.tree_settings["mode"] == "classify":
            from scipy.stats import mode
            return mode(parallel_predictions, axis=0)[0].flatten()
        elif self.tree_settings["mode"] == "regression":
            return np.mean(parallel_predictions, axis=0)
