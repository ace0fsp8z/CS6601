import numpy as np
from collections import Counter
import time
import resource

class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function,class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes."""
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)

def build_decision_tree():
    """Create decision tree
    capable of handling the provided 
    data."""
    # TODO: build full tree from root
    decision_tree_root = None
    
    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    #TODO output should be [[true_positive, false_negative], [false_positive, true_negative]]
    #TODO output is a list
    raise NotImplemented()

def precision(classifier_output, true_labels):
    #TODO precision is measured as: true_positive/ (true_positive + false_positive)
    raise NotImplemented()
    
def recall(classifier_output, true_labels):
    #TODO: recall is measured as: true_positive/ (true_positive + false_negative)
    raise NotImplemented()
    
def accuracy(classifier_output, true_labels):
    #TODO accuracy is measured as:  correct_classifications / total_number_examples
    raise NotImplemented()

def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as either 0 or 1)."""
    # TODO: finish this
    raise NotImplemented()
    
def information_gain(previous_classes, current_classes ):
    """Compute the information gain between the
    previous and current classes (each 
    a list of 0 and 1 values)."""
    # TODO: finish this
    raise NotImplemented()

class DecisionTree():
    """Class for automatic tree-building
    and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with an empty root
        and the specified depth limit."""
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__()."""
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):  
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""
        #TODO: finish this
        raise NotImplemented()
        
    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        class_labels = []
        #TODO: finish this
        raise NotImplemented()
        return class_labels

def generate_k_folds(dataset, k):
    #TODO this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    raise NotImplemented()

class RandomForest():
    """Class for random forest
    classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        """Create a random forest with a fixed 
        number of trees, depth limit, example
        sub-sample rate and attribute sub-sample
        rate."""
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of 
        decision trees."""
        # TODO implement the above algorithm
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based
        on the trained random forest."""
        # TODO implement classification for a random forest.
        raise NotImplemented()

class ChallengeClassifier():
    
    def __init__(self):
        # initialize whatever parameters you may need here-
        # this method will be called without parameters 
        # so if you add any to make parameter sweeps easier, provide defaults
        raise NotImplemented()
        
    def fit(self, features, classes):
        # fit your model to the provided features
        raise NotImplemented()
        
    def classify(self, features):
        # classify each feature in features as either 0 or 1.
        raise NotImplemented()

class Vectorization():

    def load_csv(self,data_file_path, class_index):
        handle = open(data_file_path, 'r')
        contents = handle.read()
        handle.close()
        rows = contents.split('\n')
        out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

        if(class_index == -1):
            classes= map(int,  out[:,class_index])
            features = out[:,:class_index]
            return features, classes
        elif(class_index == 0):
            classes= map(int,  out[:, class_index])
            features = out[:, 1:]
            return features, classes
        else:
            return out

    # Vectorization #1: Loops!
    # This function takes one matrix, multiplies by itself and then adds to itself.
    # Output: return a numpy array
    # 1 point
    def non_vectorized_loops(self, data):
        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = data[row][col] * data[row][col] + data[row][col]
        return non_vectorized

    def vectorized_loops(self, data):
        # TODO vectorize the code from above
        # Bonnie time to beat: 0.09 seconds
        raise NotImplemented()
        
    def vectorize_1(self):
        data = self.load_csv('vectorize.csv', 1)
        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        real_answer = self.non_vectorized_loops(data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        print 'Non-vectorized code took %s seconds' % str(end_time-start_time)

        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        my_answer = self.vectorized_loops(data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        print 'Vectorized code took %s seconds' % str(end_time-start_time)
        
        assert np.array_equal(real_answer, my_answer), "TEST FAILED"
    
    # Vectorization #2: Slicing and summation
    # This function searches through the first 100 rows, looking for the row with the max sum 
    # (ie, add all the values in that row together)
    # Output: return the max sum as well as the row number for the max sum
    # 3 points
    def non_vectorized_slice(self, data):
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if (temp_sum > max_sum):
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        # TODO vectorize the code from above
        # Bonnie time to beat: 0.07 seconds
        raise NotImplemented()
        
    def vectorize_2(self):
        data = self.load_csv('vectorize.csv', 1)
        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        real_sum, real_sum_index = self.non_vectorized_slice(data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        print 'Non-vectorized code took %s seconds' % str(end_time-start_time)

        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        my_sum, my_sum_index = self.vectorized_slice(data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        print 'Vectorized code took %s seconds' % str(end_time-start_time)

        assert (real_sum==my_sum),"TEST FAILED"
        assert (real_sum_index==my_sum_index), "TEST FAILED"
        
    # Vectorization #3: Flattening and dictionaries 
    # This function flattens down data into a 1d array, creates a dictionary of how often a
    # positive number appears in the data and displays that value
    # Output: list of tuples [(1203,3)] = 1203 appeared 3 times in data
    # 3 points
    def non_vectorized_flatten(self, data):
        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        # TODO vectorize the code from above
        # Bonnie time to beat: 15 seconds
        raise NotImplemented()

    def vectorize_3(self):
        data = self.load_csv('vectorize.csv', 1)
        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        answer_unique = self.non_vectorized_flatten(data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        print 'Non-vectorized code took %s seconds'% str(end_time-start_time)

        start_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        my_unique = self.vectorized_flatten(data)
        end_time = resource.getrusage(resource.RUSAGE_SELF).ru_utime * 1000
        print 'Vectorized code took %s seconds'% str(end_time-start_time)

        assert np.array_equal(answer_unique, my_unique), "TEST FAILED"