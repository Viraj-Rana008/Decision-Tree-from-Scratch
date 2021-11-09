from numpy.core.fromnumeric import nonzero
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r'E:\Aditya Rana\Documents\Machine Learning\Decision Tree\wifi_localization.csv')

class DecisionTreeClasifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def best_split(self, X, y):
        
        # Need atleast 2 nodes to split
        m = y.size
        if m<=1:
            return None, None

        num_parent = [np.sum(y==c) for c in range(self.num_of_classes)]

        best_gini = 1.0 - sum((n/m)**2 for n in num_parent)
        best_ind, best_thr = None, None

        for ind in range(self.num_of_features):
            # Sort along feature[ind]
            thresholds, classes = zip(*sorted(zip(X[:, ind], y)))
            
            num_left  = [0] * self.num_of_classes
            num_right = num_parent.copy()
            for i in range(1,m):
                c = classes[i-1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left  = 1.0 - sum((num_right[x] / i)**2 for x in range(self.num_of_classes))
                gini_right = 1.0 - sum((num_right[x] / (m-i)**2 for x in range(self.num_of_classes)))

                gini = (i * gini_left + (m-i) * gini_right) / m 

                # Not to split two points of same feature value in different side
                if thresholds[i] == thresholds[i-1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_ind  = ind
                    best_thr  = (thresholds[i] + thresholds[i-1]) / 2  #setting threshold as midpoint
        
        return best_ind, best_thr

    def fit(self, X, Y):
        self.num_of_classes = len(set(Y))
        self.num_of_features = X.shape[1]
        self.root = self.build_tree(X,Y)
    
    def build_tree(self, X, Y, depth=0):

class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class) -> None:
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

