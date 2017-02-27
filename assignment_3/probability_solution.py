# Yonathan Lim
# yhalim3
# CS6601
# Assignment 3

"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
import numpy as np
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    T_node = BayesNode(0, 2, name='temperature')
    G_node = BayesNode(1, 2, name='gauge')
    F_G_node = BayesNode(2, 2, name='faulty gauge')
    F_A_node = BayesNode(3, 2, name='faulty alarm')
    A_node = BayesNode(4, 2, name='alarm')

    T_node.add_child(G_node)
    T_node.add_child(F_G_node)

    F_G_node.add_parent(T_node)
    F_G_node.add_child(G_node)

    G_node.add_parent(T_node)
    G_node.add_parent(F_G_node)
    G_node.add_child(A_node)

    F_A_node.add_child(A_node)

    A_node.add_parent(G_node)
    A_node.add_parent(F_A_node)

    nodes = [T_node, G_node, F_G_node, F_A_node, A_node]

    return BayesNet(nodes)

def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    # 1
    dist = np.zeros([F_G_node.size(), T_node.size(), G_node.size()], dtype=np.float32)
    dist[0, 0, :] = [0.05, 0.95]
    dist[0, 1, :] = [0.05, 0.95]
    dist[1, 0, :] = [0.8, 0.2]
    dist[1, 1, :] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes=[F_G_node, T_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    # 2
    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([], [])
    F_A_distribution[index] = [0.85, 0.15]
    F_A_node.set_dist(F_A_distribution)

    # 3
    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([], [])
    T_distribution[index] = [0.8, 0.2]
    T_node.set_dist(T_distribution)

    # 4
    dist = np.zeros([T_node.size(), F_G_node.size()], dtype=np.float32)
    dist[0, :] = [0.95, 0.05]
    dist[1, :] = [0.2, 0.8]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)

    # 5
    dist = np.zeros([F_A_node.size(), G_node.size(), A_node.size()], dtype=np.float32)
    dist[0, 0, :] = [0.1, 0.9]
    dist[0, 1, :] = [0.1, 0.9]
    dist[1, 0, :] = [0.45, 0.55]
    dist[1, 1, :] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[F_A_node, G_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    return bayes_net

def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings], range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot], range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    T_node = bayes_net.get_node_by_name('temperature')
    A_node = bayes_net.get_node_by_name('alarm')
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot], range(Q.nDims))
    temp_prob = Q[index]
    return temp_prob


def set_skill_distribution(node):
    distribution = DiscreteDistribution(node)
    index = distribution.generate_index([], [])
    distribution[index] = [0.15, 0.45, 0.3, 0.1]
    node.set_dist(distribution)
    return node


def set_skill_diff_distribution(node, dist_nodes):
    dist = np.zeros([dist_node.size() for dist_node in dist_nodes], dtype=np.float32)
    # skill diff = 0
    dist[0, 0] = dist[1, 1] = dist[2, 2] = dist[3, 3] = [0.1, 0.1, 0.8]

    # skill diff = 1
    dist[0, 1] = dist[1, 2] = dist[2, 3] = [0.2, 0.6, 0.2]
    dist[1, 0] = dist[2, 1] = dist[3, 2] = [0.6, 0.2, 0.2]

    # skill diff = 2
    dist[0, 2] = dist[1, 3] = [0.15, 0.75, 0.1]
    dist[2, 0] = dist[3, 1] = [0.75, 0.15, 0.1]

    # skill diff = 3
    dist[0, 3] = [0.05, 0.9, 0.05]
    dist[3, 0] = [0.9, 0.05, 0.05]

    distribution = ConditionalDiscreteDistribution(nodes=dist_nodes, table=dist)
    node.set_dist(distribution)
    return node


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    A_node = BayesNode(0, 4, name='A')
    B_node = BayesNode(1, 4, name='B')
    C_node = BayesNode(2, 4, name='C')
    AvB_node = BayesNode(3, 3, name='AvB')
    BvC_node = BayesNode(4, 3, name='BvC')
    CvA_node = BayesNode(5, 3, name='CvA')

    A_node.add_child(AvB_node)
    A_node.add_child(CvA_node)

    B_node.add_child(AvB_node)
    B_node.add_child(BvC_node)

    C_node.add_child(BvC_node)
    C_node.add_child(CvA_node)

    AvB_node.add_parent(A_node)
    AvB_node.add_parent(B_node)

    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)

    CvA_node.add_parent(A_node)
    CvA_node.add_parent(C_node)

    A_node = set_skill_distribution(A_node)
    B_node = set_skill_distribution(B_node)
    C_node = set_skill_distribution(C_node)

    AvB_node = set_skill_diff_distribution(AvB_node, [A_node, B_node, AvB_node])
    BvC_node = set_skill_diff_distribution(BvC_node, [B_node, C_node, BvC_node])
    CvA_node = set_skill_diff_distribution(CvA_node, [C_node, A_node, CvA_node])

    nodes = [A_node, B_node, C_node, AvB_node, BvC_node, CvA_node]

    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0, 0, 0]
    AvB_node = bayes_net.get_node_by_name('AvB')
    CvA_node = bayes_net.get_node_by_name('CvA')
    BvC_node = bayes_net.get_node_by_name('BvC')

    engine = EnumerationEngine(bayes_net)
    engine.evidence[AvB_node] = 0
    engine.evidence[CvA_node] = 2
    Q = engine.marginal(BvC_node)[0]

    match_results = [0, 1, 2]
    for i, result in enumerate(match_results):
        index = Q.generate_index([result], range(Q.nDims))
        posterior[i] = Q[index]

    return posterior  # list


def get_random_state(nodes):
    return [np.random.choice(xrange(node.numValues)) for node in nodes]


def get_initial_state(nodes, initial_state):
    # if initial_state is not defined, get random initial
    if initial_state is None or len(initial_state) != len(nodes):
        return get_random_state(nodes)
    return initial_state


def get_markov_blanket_probability(nodes, node, state):
    # markov blanket includes its parent, children, and the other parents
    p = get_probability(nodes, node, state)
    for child in node.children:
        p = p * get_probability(nodes, child, state)
    return p


def get_probability(nodes, node, state):
    dist = node.dist.table
    # has parents means node is match
    if len(node.parents) > 0:
        values = []
        for name in [node.name[0], node.name[2]]:
            for parent in node.parents:
                if name == parent.name:
                    value = state[nodes.index(parent)]
                    values.append(value)
        # find the probabilities given the state values
        dist = dist[values[0], values[1]]
    # return the probability given the node value
    p = dist[state[nodes.index(node)]]
    return p


def Gibbs_sampler(bayes_net, initial_state, evidence_map = {}):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    nodes = list(bayes_net.nodes)
    initial_state = get_initial_state(nodes, initial_state)

    # get random node to change
    node = None
    while node is None or node in evidence_map:
        node = nodes[np.random.randint(len(nodes))]

    sample = list(initial_state)
    index = nodes.index(node)
    values = xrange(node.numValues)

    # get conditional probabilities distribution
    p_list = []
    for value in values:
        sample[index] = value
        p_list.append(get_markov_blanket_probability(nodes, node, sample))

    # normalize p_list
    p_list = np.array(p_list) / np.sum(p_list)

    # sample based on conditional probabilities
    value = np.random.choice(values, p=p_list)
    sample[index] = value

    return tuple(sample)


def MH_sampler(bayes_net, initial_state, evidence_map = {}):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    nodes = list(bayes_net.nodes)
    initial_state = get_initial_state(nodes, initial_state)

    sample = list(initial_state)
    candidate = list(initial_state)

    filtered_nodes = [node for node in nodes if node not in evidence_map]

    # propose a candidate
    for node in filtered_nodes:
        index = nodes.index(node)
        value = np.random.choice(node.numValues)
        candidate[index] = value

    # get acceptance probability
    current_probability = 1.
    candidate_probability = 1.
    for node in nodes:
        current_probability *= get_probability(nodes, node, initial_state)
        candidate_probability *= get_probability(nodes, node, candidate)

    if candidate_probability > current_probability:
        # If probability of new state is greater than that of the old state, you accept the candidate.
        sample = candidate
    else:
        # Else, you accept/reject the candidate by randomly choosing a value between 0 and 1 (excluding 0 and 1).
        alpha = candidate_probability / current_probability
        if alpha > np.random.uniform(0, 1):
            sample = candidate

    return tuple(sample)


def get_posterior(sample_count):
    return np.array(sample_count, dtype=np.float32) / np.sum(sample_count)


def is_accepted(current_posterior, prev_posterior, delta):
    if prev_posterior is None:
        return False
    current_posterior = np.array(current_posterior)
    prev_posterior = np.array(prev_posterior)
    diff = np.array(np.abs(current_posterior - prev_posterior))
    return np.any(current_posterior != prev_posterior) and np.all(diff < delta)


def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0, 0, 0]  # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0, 0, 0]  # posterior distribution of the BvC match as produced by MH

    N = 100000
    BURN_IN = 10000
    CONSECUTIVE_THRESHOLD = 10
    nodes = list(bayes_net.nodes)
    AvB = bayes_net.get_node_by_name('AvB')
    BvC = bayes_net.get_node_by_name('BvC')
    CvA = bayes_net.get_node_by_name('CvA')

    BvC_index = nodes.index(BvC)
    evidence_map = {AvB: 0, CvA: 2}

    initial_state = get_initial_state(nodes, initial_state)

    current_state = list(initial_state)
    for node, value in evidence_map.iteritems():
        current_state[nodes.index(node)] = value

    sample_count = [0, 0, 0]
    prev_posterior = None
    consecutive_iteration = 0
    for Gibbs_count in xrange(N):
        next_state = Gibbs_sampler(bayes_net, current_state, evidence_map)
        current_state = next_state
        sample_count[next_state[BvC_index]] += 1

        if Gibbs_count < BURN_IN:
            continue

        Gibbs_convergence = get_posterior(sample_count)

        if is_accepted(Gibbs_convergence, prev_posterior, delta):
            consecutive_iteration += 1
        else:
            consecutive_iteration = 0
        prev_posterior = Gibbs_convergence

        if consecutive_iteration >= CONSECUTIVE_THRESHOLD:
            break

    current_state = list(initial_state)
    sample_count = [0, 0, 0]
    prev_posterior = None
    consecutive_iteration = 0
    for MH_count in xrange(N):
        next_state = MH_sampler(bayes_net, current_state, evidence_map)

        if np.all(np.array(current_state) == np.array(next_state)):
            MH_rejection_count += 1

        current_state = next_state
        sample_count[next_state[BvC_index]] += 1

        if MH_count < BURN_IN:
            continue

        MH_convergence = get_posterior(sample_count)

        if is_accepted(MH_convergence, prev_posterior, delta):
            consecutive_iteration += 1
        else:
            consecutive_iteration = 0
        prev_posterior = MH_convergence

        if consecutive_iteration >= CONSECUTIVE_THRESHOLD:
            break

    return list(Gibbs_convergence), list(MH_convergence), Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 0
    options = ['Gibbs', 'Metropolis-Hastings']
    factor = 1
    return options[choice], factor
