import numpy as np 
import pandas as pf 
from collections import Counter

class decision_tree():

    def __init__(self, min_information_gain, min_samples_leaf, max_depth):
        self.min_information_gain = min_information_gain
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.tree = None
    
    def label_probabilities(self, labels:list) ->dict:
        total_count  = len(labels)
        label_count = Counter(labels)

        label_probs = {label: count / total_count for label, count in label_count.items()}

        return label_probs



    def calculate_entropy(self, labels:list) ->float:
        label_probs  =self.label_probabilities(labels)
        entropy = sum([-p * np.log2(p) for p in label_probs.values() if p >0])
        return entropy 
    


    
    # def class_probabilities(self, labels:list) ->list: #I guess we won't need this one, let's see
    #     total_count = len(labels)

    #     probs =  [label_count/total_count for label_count in Counter(labels).values()]
    #     return(probs)
    
    def partition_entropy(self, subsets:list) ->float:
        t_c = sum(len(s) for s in subsets)
        partition_entropy = sum(self.calculate_entropy(s) * len(s)/t_c for s in subsets)
        return partition_entropy
   
    def calculate_information_gain(self, parent_labels:list, left_labels:list, right_labels:list) ->float:
        parent_entropy = self.calculate_entropy(parent_labels)
        t_c = len(parent_labels)

        child_entropy = self.partition_entropy([left_labels, right_labels])

        info_gain = parent_entropy - child_entropy

        return info_gain
    
    def find_best_split(self,X,y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_features = len(X[0])

        for feature_index in range(n_features): #looping through features
            values = [sample[feature_index] for sample in X]

            thresholds = set(values) 
            # [2,3,4,2]

            for threshold in thresholds:
                left_y, right_y = [],[]

                for i, sample in enumerate(X):
                    if sample[feature_index] ==threshold:
                        left_y.append(y[i]) 
                    else:
                        right_y.append(y[i]) 

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    print(f"Skipping threshold - {threshold} as the split is too small.")
                    continue

                #calculate information gain for splits
                gain = self.calculate_information_gain(y, left_y, right_y)
                
                if gain > best_gain and gain > self.min_information_gain:
                    best_gain = gain 
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_gain, best_threshold

        #build tree
    def build_tree(self, X, y, depth = 0):
        #best cases for stopping criteria
        print(f"Building tree at depth {depth}, len(X)={len(X)}, len(y)={len(y)}")

        if len(set(y)) == 1: 
            print(f"all labels are same {y[0]}")
            return {"label":y[0]}

        if len(X) < self.min_samples_leaf:
            majority = self.majority_class(y)
            print(f"too few samples, retuning majority class {majority}")
            return{'label': majority}

        if depth ==self.max_depth:
            majority =self.majority_class(y)
            print(f"reached max depth, returning majority class {majority}")
            return{'label': majority}

        best_feature, best_gain, best_threshold = self.find_best_split(X,y)

        if best_gain < self.min_information_gain:
            majority = self.majority_class(y)
            print(f"too small information gain, returning majority class {majority}")
            return {"label": majority}

        #split data based on feature and threshold
        left_X, left_y, right_X, right_y =self.split_data(X,y,best_feature, best_threshold)
        print(f"splitting on feature {best_feature}, threshold {best_threshold}")
        #recurively build the left and right subtrees
        left_tree = self.build_tree(left_X, left_y, depth +1)
        right_tree =  self.build_tree(right_X, right_y, depth + 1)

        self.tree =  {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}
        return self.tree


        

    
    def majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def split_data(self, X, y, feature_index, threshold):
        left_X, left_y, right_X, right_y =[],[],[],[]

        for i, sample in enumerate(X):
            if sample[feature_index] == threshold:
                left_X.append(sample)
                left_y.append(y[i])
            else:
                right_X.append(sample)
                right_y.append(y[i])
        
        return left_X, left_y, right_X, right_y

    def predict(self, sample, tree =None):
        if tree is None:
            tree = self.tree
        if "label" in tree:
            return tree["label"]
        if sample[tree["feature"]] <=tree["threshold"]:
            return self.predict(sample, tree["left"])
        else:
            return self.predict(sample, tree["right"])
X = [
    [2, 3],
    [1, 2],
    [5, 3],
    [7, 6],
    [6, 4]
]
y = ['A', 'A', 'B', 'B', 'B']

tree = decision_tree(min_information_gain =0.1, min_samples_leaf=1, max_depth = 3)
tree.build_tree(X,y)


test_samples = [[4, 3], [6, 5], [1, 2]]
for sample in test_samples:
    prediction = tree.predict(sample)
    print(f"Prediction for {sample}: {prediction}")


