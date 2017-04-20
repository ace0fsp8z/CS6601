# This file is your main submission that will be graded against. Only copy-paste
# code on the relevant classes included here from the IPython notebook. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.


def part_1_a(): #(10 pts)
    # TODO: Fill below!
    B_transition_probs = {
        
    }

    B_emission_probs = {
       
    }
    
    B_prior = {
       
    }
    
    C_transition_probs = {
       
    }

    C_emission_probs = {
       
    }
    
    C_prior = {
       
    }
    
    return B_prior, B_transition_probs, B_emission_probs, C_prior, C_transition_probs, C_emission_probs

import numpy as np
def viterbi(evidence_vector, prior, states, transition_probs, emission_probs):  #(45 pts)
    sequence=[]
    probability=0
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

 
    return sequence, probability
   

def part_2_a(): 
    #TO DO: fill in below
    states=(
    )
    
    prior_probs = {
       
    }
    
    transition_probs={
        
    }
        
    emission_probs={
        
    }
         
    return states, prior_probs, transition_probs, emission_probs


def quick_check():
    #TO DO: fill the probabilities, 5 points
    
    #prior probability for C1
    prior_C1=
    
    #transition probability from A3 to L1
    A3_L1=
    
    #transition probability from B4 to B5
    B4_B5=
    
    #transition probability from W1 to B1
    W1_B1=
    
    #transition probability from L1 to L1
    L1_L1=
    
    return prior_C1,A3_L1,B4_B5,W1_B1,L1_L1
    

def part_2_b(evidence_vector, prior, states, transition_probs, emission_probs):
    sequence=''
    probability=0
    '''
    TO DO: fill this (40 points)
    Output:
        sequence: a string of most likely decoded letter sequence (like 'A B A CAC', using uppercase)
        probability: float
    '''
        
    return sequence, probability
