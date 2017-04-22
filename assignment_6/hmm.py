# Yonathan Lim
# yhalim3
# CS6601
# Assignment 6

# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.


def part_1_a(): #(10 pts)
    # TODO: Fill below!
    B_transition_probs = {
        'B1': {'B1': .667, 'B2': .333, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0},
        'B2': {'B1': 0, 'B2': 0, 'B3': 1, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0},
        'B3': {'B1': 0, 'B2': 0, 'B3': 0, 'B4': 1, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0},
        'B4': {'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 1, 'B6': 0, 'B7': 0, 'Bend': 0},
        'B5': {'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 1, 'B7': 0, 'Bend': 0},
        'B6': {'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 1, 'Bend': 0},
        'B7': {'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 1},
        'Bend': {'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 1}
    }

    B_emission_probs = {
        'B1': [0, 1],
        'B2': [1, 0],
        'B3': [0, 1],
        'B4': [1, 0],
        'B5': [0, 1],
        'B6': [1, 0],
        'B7': [0, 1],
        'Bend': [0, 0]
    }

    B_prior = {
        'B1': 1,
        'B2': 0,
        'B3': 0,
        'B4': 0,
        'B5': 0,
        'B6': 0,
        'B7': 0,
        'Bend': 0
    }

    C_transition_probs = {
        'C1': {'C1': .667, 'C2': .333, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0},
        'C2': {'C1': 0, 'C2': 0, 'C3': 1, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0},
        'C3': {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 1, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0},
        'C4': {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 1, 'C6': 0, 'C7': 0, 'Cend': 0},
        'C5': {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': .667, 'C6': .333, 'C7': 0, 'Cend': 0},
        'C6': {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 1, 'Cend': 0},
        'C7': {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 1},
        'Cend': {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 1}
    }

    C_emission_probs = {
        'C1': [0, 1],
        'C2': [1, 0],
        'C3': [0, 1],
        'C4': [1, 0],
        'C5': [0, 1],
        'C6': [1, 0],
        'C7': [0, 1],
        'Cend': [0, 0]
    }

    C_prior = {
        'C1': 1,
        'C2': 0,
        'C3': 0,
        'C4': 0,
        'C5': 0,
        'C6': 0,
        'C7': 0,
        'Cend': 0
    }
    
    return B_prior, B_transition_probs, B_emission_probs, C_prior, C_transition_probs, C_emission_probs

import numpy as np
def viterbi(evidence_vector, prior, states, transition_probs, emission_probs):  #(45 pts)
    """
        
        Input:
    
            evidence_vector: A list of dictionaries mapping evidence variables to their values

            prior: A dictionary corresponding to the prior distribution over states

            states: A list of all possible system states

            transition_probs: A dictionary mapping states onto dictionaries mapping states onto probabilities

            emission_probs: A dictionary mapping states onto dictionaries mapping evidence variables onto 
                        probabilities for their possible values

                    
        Output:
            sequence: A list of states that is the most likely sequence of states explaining the evidence, like 
            ['A1', 'A2', 'A3', 'A3', 'A3']
            probability: float
        
    """
    # pseudocode from https://en.wikipedia.org/wiki/Viterbi_algorithm modified to use log probability
    V = [{}]
    for state in states:
        prob = prior[state] * emission_probs[state][evidence_vector[0]]
        V[0][state] = {
            'logprob': np.log(prob) if prob else float('-inf'),
            'prev': None
        }

    # run viterbi for t > 0
    for t in range(1, len(evidence_vector)):
        V.append({})
        for state in states:
            prob = np.zeros(len(states))
            for i, prev_state in enumerate(states):
                if V[t - 1][prev_state]['logprob'] == float('-inf'):
                    prob[i] = float('-inf')
                else:
                    p = transition_probs[prev_state][state] * emission_probs[state][evidence_vector[t]]
                    prob[i] = V[t - 1][prev_state]['logprob'] + np.log(p) if p else float('-inf')
            idx = np.argmax(prob)
            prev_state = states[idx]
            V[t][state] = {'logprob': prob[idx], 'prev': prev_state}

    sequence = []
    # the highest probability
    max_logprob = max(value['logprob'] for value in V[-1].values())
    if max_logprob != float('-inf'):
        previous = None
        # get most probable state and its backtrack
        for state, data in V[-1].items():
            if data['logprob'] == max_logprob:
                sequence.append(state)
                previous = state
                break
        # follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            sequence.insert(0, V[t + 1][previous]['prev'])
            previous = V[t + 1][previous]['prev']

    return sequence, round(np.exp(max_logprob), 3)
   

def part_2_a(): 
    #TO DO: fill in below
    states = (
        'A1', 'A2', 'A3', 'Aend',
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Bend',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'Cend',
        'L1', 'Lend', 'W1', 'Wend'
    )

    prior_probs = {
        'A1': .333, 'A2': 0, 'A3': 0, 'Aend': 0,
        'B1': .333, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
        'C1': .333, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
        'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
    }

    transition_probs = {
        'A1': {
            'A1': .2, 'A2': .8, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'A2': {
            'A1': 0, 'A2': .2, 'A3': .8, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'A3': {
            'A1': 0, 'A2': 0, 'A3': .667, 'Aend': .111,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': .111, 'Lend': 0, 'W1': .111, 'Wend': 0
        },
        'Aend': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 1,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B1': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': .667, 'B2': .333, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B2': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': .2, 'B3': .8, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B3': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': .2, 'B4': .8, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B4': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': .2, 'B5': .8, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B5': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': .2, 'B6': .8, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B6': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': .2, 'B7': .8, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'B7': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': .2, 'Bend': .267,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': .267, 'Lend': 0, 'W1': .267, 'Wend': 0
        },
        'Bend': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 1,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C1': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': .667, 'C2': .333, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C2': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': .2, 'C3': .8, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C3': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': .2, 'C4': .8, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C4': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': .2, 'C5': .8, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C5': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': .667, 'C6': .333, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C6': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': .2, 'C7': .8, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'C7': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': .2, 'Cend': .267,
            'L1': .267, 'Lend': 0, 'W1': .267, 'Wend': 0
        },
        'Cend': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 1,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        # 'L1': {
        #     'A1': .083, 'A2': 0, 'A3': 0, 'Aend': 0,
        #     'B1': .083, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
        #     'C1': .083, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
        #     'L1': .667, 'Lend': .083, 'W1': 0, 'Wend': 0
        # },
        'L1': {
            'A1': .111, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': .111, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': .111, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': .667, 'Lend': 0, 'W1': 0, 'Wend': 0
        },
        'Lend': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 1, 'W1': 0, 'Wend': 0
        },
        # 'W1': {
        #     'A1': .036, 'A2': 0, 'A3': 0, 'Aend': 0,
        #     'B1': .036, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
        #     'C1': .036, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
        #     'L1': 0, 'Lend': 0, 'W1': .857, 'Wend': .036
        # },
        'W1': {
            'A1': .048, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': .048, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': .048, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': .857, 'Wend': 0
        },
        'Wend': {
            'A1': 0, 'A2': 0, 'A3': 0, 'Aend': 0,
            'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0, 'B6': 0, 'B7': 0, 'Bend': 0,
            'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0, 'C6': 0, 'C7': 0, 'Cend': 0,
            'L1': 0, 'Lend': 0, 'W1': 0, 'Wend': 1
        }
    }

    emission_probs = {
        'A1': [.2, .8],
        'A2': [.8, .2],
        'A3': [.2, .8],
        'Aend': [0, 0],
        'B1': [.2, .8],
        'B2': [.8, .2],
        'B3': [.2, .8],
        'B4': [.8, .2],
        'B5': [.2, .8],
        'B6': [.8, .2],
        'B7': [.2, .8],
        'Bend': [0, 0],
        'C1': [.2, .8],
        'C2': [.8, .2],
        'C3': [.2, .8],
        'C4': [.8, .2],
        'C5': [.2, .8],
        'C6': [.8, .2],
        'C7': [.2, .8],
        'Cend': [0, 0],
        'L1': [.8, .2],
        'Lend': [0, 0],
        'W1': [.8, .2],
        'Wend': [0, 0]
    }
         
    return states, prior_probs, transition_probs, emission_probs


def quick_check():
    #TO DO: fill the probabilities, 5 points
    
    #prior probability for C1
    prior_C1=.333

    #transition probability from A3 to L1
    A3_L1=.111

    #transition probability from B4 to B5
    B4_B5=.8

    #transition probability from W1 to B1
    W1_B1=.036

    #transition probability from L1 to L1
    L1_L1=.083
    
    return prior_C1,A3_L1,B4_B5,W1_B1,L1_L1
    

def part_2_b(evidence_vector, prior, states, transition_probs, emission_probs):
    """
    TO DO: fill this (40 points)
    Output:
        sequence: a string of most likely decoded letter sequence (like 'A B A CAC', using uppercase)
        probability: float
    """
    sequence = ''
    observed_sequence, probability = viterbi(evidence_vector, prior, states, transition_probs, emission_probs)

    prev_seq = None
    for i in xrange(len(observed_sequence)):
        seq = observed_sequence[i]
        if prev_seq is None or seq[0] != prev_seq[0]:
            prev_seq = seq
            if seq.startswith('W'):
                sequence += ' '
            elif seq.startswith('L'):
                pass
            else:
                sequence += seq[0]

    return sequence, probability
