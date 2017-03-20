import numpy as np
from collections import Counter
import time
import resource

class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
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
    decision_tree_root = DecisionNode(None, None, lambda feature: feature[0] == 1)
    decision_tree_root.left = DecisionNode(None, None, None, 1)
    decision_tree_root.right = DecisionNode(None, None, lambda feature: feature[2] == 1)

    a3 = decision_tree_root.right
    a3.left = DecisionNode(None, None, lambda feature: feature[3] == 1)
    a3.right = DecisionNode(None, None, lambda feature: feature[3] == 0)

    a4_1 = a3.left
    a4_1.left = DecisionNode(None, None, None, 1)
    a4_1.right = DecisionNode(None, None, None, 0)

    a4_2 = a3.right
    a4_2.left = DecisionNode(None, None, None, 1)
    a4_2.right = DecisionNode(None, None, None, 0)

    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    #TODO output should be [[true_positive, false_negative], [false_positive, true_negative]]
    #TODO output is a list
    output = [
        [0, 0],
        [0, 0]
    ]
    for i, actual in enumerate(true_labels):
        prediction = classifier_output[i]
        if actual == 1:
            if prediction == actual:
                output[0][0] += 1
            else:
                output[0][1] += 1
        elif actual == 0:
            if prediction == actual:
                output[1][1] += 1
            else:
                output[1][0] += 1
    return output


def precision(classifier_output, true_labels):
    #TODO precision is measured as: true_positive/ (true_positive + false_positive)
    [tp, _], [fp, _] = confusion_matrix(classifier_output, true_labels)
    return tp / float(tp + fp)


def recall(classifier_output, true_labels):
    #TODO: recall is measured as: true_positive/ (true_positive + false_negative)
    [tp, fn], _ = confusion_matrix(classifier_output, true_labels)
    return tp / float(tp + fn)


def accuracy(classifier_output, true_labels):
    #TODO accuracy is measured as:  correct_classifications / total_number_examples
    [tp, fn], [fp, tn] = confusion_matrix(classifier_output, true_labels)
    return (tp + tn) / float(tp + fn + fp + tn)


def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as either 0 or 1)."""
    # TODO: finish this
    # H(x) = -Sigma [ p(x_i) log2(p(x_i)) ]
    # H(x) = -(p(x_i) log2(p(x_i)) + (1 - p(x_i)) log2(1 - p(x_i)))
    total = len(class_vector) or 1
    p = sum(class_vector) / float(total)
    x1 = p * np.log2(p) if p > 0 else 0
    x2 = (1 - p) * np.log2(1 - p) if p < 1 else 0
    H = -1 * (x1 + x2)
    return H


def information_gain(previous_classes, current_classes ):
    """Compute the information gain between the
    previous and current classes (each 
    a list of 0 and 1 values)."""
    # TODO: finish this
    # Remainder(A) = Sigma [ ((pk + nk) / (p + n)) * B(pk / (pk + nk)) ]
    # Gain(A) = B(p / (p + n) - Remainder(A))
    total = len(previous_classes)

    remainder = 0
    for current_class in current_classes:
        remainder += (len(current_class) / float(total)) * entropy(current_class)

    gain = entropy(previous_classes) - remainder
    return gain


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
        classes = np.array(classes)
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):  
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""
        #TODO: finish this
        # 1) Check for base cases:
        #      a) If all elements of a list are of the same class, return a leaf node with the appropriate class label.
        #      b) If a specified depth limit is reached, return a leaf labeled with the most frequent class.
        # 2) For each attribute alpha: evaluate the normalized information gain gained by splitting on attribute alpha
        # 3) Let alpha_best be the attribute with the highest normalized information gain
        # 4) Create a decision node that splits on alpha_best
        # 5) Recur on the sublists obtained by splitting on alpha_best, and add those nodes as children of node
        if np.all(np.array(classes) == 1):
            return DecisionNode(None, None, None, 1)
        elif np.all(np.array(classes) == 0):
            return DecisionNode(None, None, None, 0)
        if depth >= self.depth_limit or depth >= len(features):
            if sum(classes) > float(len(classes) / 2):
                return DecisionNode(None, None, None, 1)
            else:
                return DecisionNode(None, None, None, 0)

        # if set(np.unique(features)) == set([0, 1]):
        #     continue
        # else:
        #     # continuous value

        # find alpha best by using variance -2 stddev to 2 stddev
        variance_list = np.std(features, axis=0)
        alpha_best = None
        beta_gain_list = []
        beta_best_list = []
        # for each feature, we want to find the best threshold
        for i in xrange(features.shape[1]):
            alpha = features[:, i]
            mean = np.mean(alpha)
            variance = variance_list[i]
            beta_best = None
            beta_gain = np.float('-inf')
            # check for -2, -1, 0, 1, 2 std dev for threshold
            for stddev in xrange(-2, 3):
                beta = alpha.copy()
                threshold = mean + stddev * variance
                indices = beta > threshold
                not_indices = np.negative(indices)
                beta[:] = 0
                beta[indices] = 1
                gain = information_gain(classes, [classes[indices], classes[not_indices]])
                if gain > beta_gain:
                    beta_gain = gain
                    beta_best = stddev
            beta_gain_list.append(beta_gain)
            beta_best_list.append(beta_best)

        alpha_best = np.argmax(beta_gain_list)
        alpha = features[:, alpha_best]
        threshold = np.mean(alpha) + beta_best_list[alpha_best] * variance_list[alpha_best]
        indices = alpha > threshold
        not_indices = np.negative(indices)

        left_features = features[indices]
        right_features = features[not_indices]
        left_classes = classes[indices]
        right_classes = classes[not_indices]
        left_node = self.__build_tree__(left_features, left_classes, depth + 1)
        right_node = self.__build_tree__(right_features, right_classes, depth + 1)

        return DecisionNode(left_node, right_node, lambda features: features[alpha_best] > threshold)
        
    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        #TODO: finish this
        class_labels = [self.root.decide(feature) for feature in features]
        return class_labels


def generate_k_folds(dataset, k):
    #TODO this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    features, classes = dataset
    classes = np.array(classes)

    indices = np.arange(classes.size)
    training_size = int(.9 * classes.size)

    folds = []
    np.random.shuffle(indices)

    for i in xrange(k):
        np.random.shuffle(indices)

        randomized_features = features[indices]
        randomized_classes = classes[indices]

        training_features = randomized_features[:training_size]
        training_classes = randomized_classes[:training_size]
        training_set = (training_features, training_classes)

        test_features = randomized_features[training_size:]
        test_classes = randomized_classes[training_size:]
        test_set = (test_features, test_classes)

        folds.append((training_set, test_set))

    return folds


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
        sample_size, attr_size = features.shape
        sample_indices = np.arange(sample_size)
        attr_indices = np.arange(attr_size)
        subsample_size = np.int(sample_size * self.example_subsample_rate)
        subattr_size = np.int(attr_size * self.attr_subsample_rate)

        for i in xrange(self.num_trees):
            subsample_indices = np.random.choice(sample_indices, subsample_size, replace=False)
            subattr_indices = np.sort(np.random.choice(attr_indices, subattr_size, replace=False))

            subsample_features = features[subsample_indices][:, subattr_indices]
            subsample_classes = classes[subsample_indices]

            tree = DecisionTree(self.depth_limit)
            tree.fit(subsample_features, subsample_classes)

            self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based
        on the trained random forest."""
        # TODO implement classification for a random forest.
        classes = np.zeros(features.shape[0])
        for tree in self.trees:
            classes = np.add(classes, tree.classify(features))
        threshold = len(self.trees) / 2.
        indices = classes >= threshold
        classes[:] = 0
        classes[indices] = 1
        return classes


class ChallengeClassifier():
    
    def __init__(self):
        # initialize whatever parameters you may need here-
        # this method will be called without parameters 
        # so if you add any to make parameter sweeps easier, provide defaults
        self.tree = RandomForest(10, 10, 0.25, 1)
        
    def fit(self, features, classes):
        # fit your model to the provided features
        self.tree.fit(features, classes)
        
    def classify(self, features):
        # classify each feature in features as either 0 or 1.
        return self.tree.classify(features)


class Vectorization():

    def load_csv(self, data_file_path, class_index):
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
        return data * data + data
        
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
        sum_list = np.sum(data[:100], axis=1)
        max_sum_index = np.argmax(sum_list)
        max_sum = sum_list[max_sum_index]
        return max_sum, max_sum_index
        
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
        flattened = data[data > 0.]
        unique_dict = Counter(flattened)
        return unique_dict.items()

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
