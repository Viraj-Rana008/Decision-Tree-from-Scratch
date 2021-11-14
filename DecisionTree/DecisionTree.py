import numpy as np
import pandas as pd

data = pd.read_csv(r'E:\Aditya Rana\Documents\wifi_localization.csv', delimiter="\t")

class Node:
    def __init__(self, features_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = features_index
        self.threshold     = threshold
        self.left          = left
        self.right         = right
        self.info_gain     = info_gain

        # For Leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None

        # Stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth         = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)

        # split until stopping condition
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            
            best_split = self.best_split(dataset, num_samples, num_features)

            # If spliting this node gives info. gain
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)

                node = Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])
                return node
        
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def best_split(self, dataset, num_smaples, num_features):
        '''
            @param:
                Dataset, num_samples = Dateset.shape[0], num_feature = Dataset.shape[1]
            @return:
                Split info. dictionary
            
            - Finds which feature and which feature value gives best info. gain
        '''
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            # loop over all unique feature value
            for threshold in possible_thresholds:
                # current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                # if split give 2 child
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    # compute info. gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    
                    # update cur_info_gain
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"]     = feature_index
                        best_split["threshold"]          = threshold
                        best_split["dataset_left"]      = dataset_left
                        best_split["dataset_right"]     = dataset_right
                        best_split["info_gain"]         = curr_info_gain
                        max_info_gain   =   curr_info_gain
        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        '''
            Split into two halves, with 'threshold' for (feature_index)th feature
        '''
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])

        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])

        return dataset_left, dataset_right
    
    def information_gain(self, parent, left_child, right_child, mode="entropy"):
        '''
            @param: 
                Parent labels, Left Child labels, Right Child Labels    (Parent -> Left + Right)
            @return:
                Infomation Gain
        '''
        weight_left     = len(left_child) / len(parent)
        weight_right    = len(right_child) / len(parent)

        if mode=="gini":
            gain = self.gini_index(parent) - (weight_left * self.gini_index(left_child) + weight_right * self.gini_index(right_child))
        else:
            gain = self.entropy(parent) - (weight_left * self.entropy(left_child) + weight_right * self.entropy(right_child))
        
        return gain
    
    def entropy(self, y):
        '''
            @param:
                Labels/Output
            @retrun:
                Entropy =  sum(-(p*log(p)))
        '''
        class_labels = np.unique(y)
        entropy = 0                

        # loop over all unique labels
        for label in class_labels:
            label_probab = len(y[y==label]) / len(y)
            entropy += -label_probab * np.log2(label_probab)
        
        return entropy

    def gini_index(self, y):
        '''
            @param:
                Labels/Output
            @return:
                Gini Index = 1 - sum(p**2)
        '''
        class_labels = np.unique(y)
        gini = 0                   

        # loop over all unique lables
        for label in class_labels:
            label_probab = len(y[y==label]) / len(y)
            gini += label_probab**2
        
        return 1-gini
    
    def calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)
    
    def fit(self, X, Y):
        '''
            @param:
                X_train, Y_train
        '''
        dataset = np.concatenate((X,Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        '''
            @param:
                X_test
            @return:
                Y_test
        '''
        prediction = [self.make_prediction(x, self.root) for x in X]
        return prediction
    
    def make_prediction(self, X, root):
        '''
            Tree Traversal till leaf node
        '''
        # Leaf-node / Base Case
        if root.value != None:
            return root.value
        
        feature_val = X[root.feature_index]
        if feature_val[0]=="-":
            feature_val = -1 * (int(feature_val[1:]))
        else:
            feature_val = int(feature_val)
        #print(type(feature_val), type(root.threshold))
        try:
            if feature_val <= root.threshold:
                return self.make_prediction(X, root.left)
            else:
                return self.make_prediction(X, root.right)
        except:
            return


training_set    = data.sample(frac=0.7, replace=False, random_state=0, axis=0)
test_set        = data.drop(training_set.index)

x_train = training_set.drop(training_set.columns[[7]], axis=1)
y_train = training_set.iloc[:,[7]]

x_test = test_set.drop(test_set.columns[[7]], axis=1)
#y_test = test_set[:,7]

clf = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)