from __future__ import division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import scipy as sp
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from helper_functions import image_to_matrix, matrix_to_image, flatten_image_matrix, unflatten_image_matrix, image_difference

from random import randint
from functools import reduce


def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    h, w = image_values.shape[:2]
    if initial_means is None:
        x_indices = np.random.choice(xrange(1, h), size=k, replace=False)
        y_indices = np.random.choice(xrange(1, w), size=k, replace=False)
        initial_means = image_values[x_indices, y_indices]

    means = initial_means
    updated_image_values = image_values.copy()
    while True:
        distances = []
        # find the distance between each pixel and the mean
        for i, mean in enumerate(means):
            distance = np.sum(np.square(image_values - mean), axis=2)
            distances.append(distance.T)
        # then merge k distance and get the label using minimum distance
        labels = np.argmin(np.vstack([distances]).T, axis=2)

        # calculate the new updated means
        updated_means = []
        for i in xrange(k):
            indices = labels == i
            mean = np.average(image_values[indices], axis=0)
            updated_means.append(mean)
        diff = np.linalg.norm(updated_means - means, axis=1)
        # update the means
        means = np.array(updated_means)
        # check for convergence
        if np.all(diff < 0.00001):
            break

    for i in xrange(k):
        indices = labels == i
        updated_image_values[indices] = means[i]

    return updated_image_values


def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr+=1
    else:
        conv_ctr =0

    return conv_ctr, conv_ctr > conv_ctr_cap


from random import randint
import math
from scipy.misc import logsumexp


class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.

        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        # self.image_matrix = image_matrix
        # self.num_components = num_components
        # if(means is None):
        #     self.means = [0]*num_components
        # else:
        #     self.means = means
        # self.variances = [0]*num_components
        # self.mixing_coefficients = [0]*num_components
        self.image_matrix = image_matrix
        self.num_components = num_components
        self.means = np.array([0] * num_components) if means is None else np.array(means)
        self.variances = np.array([0]*num_components)
        self.mixing_coefficients = np.array([0]*num_components)
        self.flatten_image = flatten_image_matrix(self.image_matrix)

    def get_probabilities(self, values):
        means = np.array(self.means)
        x1 = -.5 * np.log(2 * np.pi * np.square(self.variances))
        x2 = np.square(values - means) / (2 * np.square(self.variances))
        return np.exp(x1 - x2) * self.mixing_coefficients

    def joint_prob(self, val):
        """Calculate the joint
        log probability of a greyscale
        value within the image.

        params:
        val = float

        returns:
        joint_prob = float
        """
        probabilities = self.get_probabilities(val)
        return np.log(np.sum(probabilities))

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).

        NOTE: this should be called before
        train_model() in order for tests
        to execute correctly.
        """
        self.means = np.random.choice(self.flatten_image.flatten(), self.num_components)
        self.variances = np.ones(self.num_components).astype(np.float)
        self.mixing_coefficients = np.ones(self.num_components) / np.float(self.num_components)

    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and
        self.variances, plus
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.

        params:
        convergence_function = function that returns True if convergence is reached
        """
        conv_ctr = 0
        prev_likelihood = self.likelihood()
        while True:
            # E step
            probabilities = self.get_probabilities(self.flatten_image)
            gamma = probabilities / np.sum(probabilities, axis=1).reshape(probabilities.shape[0], 1)

            # M step
            N_k = np.sum(gamma, axis=0)
            self.means = np.sum(gamma * self.flatten_image, axis=0) / N_k
            self.variances = np.sum(gamma * np.power(self.flatten_image - self.means, 2), axis=0) / N_k
            self.mixing_coefficients = N_k / self.flatten_image.size

            # Evaluate log likelihood and check for convergence
            new_likelihood = self.likelihood()
            conv_ctr, converged = default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10)
            prev_likelihood = new_likelihood

            if converged:
                break

    def segment(self):
        """
        Using the trained model,
        segment the image matrix into
        the pre-specified number of
        components. Returns the original
        image matrix with the each
        pixel's intensity replaced
        with its max-likelihood
        component mean.

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        raise NotImplementedError()
        return segment

    def likelihood(self):
        """Assign a log
        likelihood to the trained
        model based on the following
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N),ln(sum((k=1 to K), mixing_k * N(x_n | mean_k, stdev_k) )))

        returns:
        log_likelihood = float [0,1]
        """
        return self.joint_prob(self.flatten_image)

    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly
        training the model and
        calculating its likelihood.
        Return the segment with the
        highest likelihood.

        params:
        iters = int

        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        raise NotImplementedError()
        return segment


class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient initializations too if that works well.]
        """
        # TODO: finish this
        raise NotImplementedError()


def new_convergence_function(previous_variables, new_variables, conv_ctr, conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    raise NotImplementedError()
    return conv_ctr, converged


class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        raise NotImplementedError()


def bayes_info_criterion(gmm):
    # TODO: finish this function
    raise NotImplementedError()
    return BIC


def BIC_likelihood_model_test():
    """Test to compare the
    models with the lowest BIC
    and the highest likelihood.

    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel
    """
    # TODO: finish this method
    raise NotImplementedError()
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563, 0.964706]
    ]
    return min_BIC_model, max_likelihood_model


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    raise NotImplementedError()
    bic = 0
    likelihood = 0
    pairs = {
        'BIC' : bic,
        'likelihood' : likelihood
    }
    return pairs


def bonus(points_array, means_array):
    """
    Return the distance from every point in points_array
    to every point in means_array.
    
    returns:
    dists = numpy array of float
    """
    # TODO: fill in the bonus function
    # REMOVE THE LINE BELOW IF ATTEMPTING BONUS
    raise NotImplementedError()
    return dists

