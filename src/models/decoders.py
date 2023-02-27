import copy
import numpy as np
class ViterbiAlgorithm:
    """
    Implement Viterbi algorithm: Takes HMM parameters from hmm.py and decode/find best path (max likelihood) of hidden state sequence
    for given observation states

    """    

    def __init__(self, hmm_object):
        """
        Initialize hmm object from hmm.py

        Args:
            hmm_object (_type_): hmm object from HiddenMarkovModel
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """
        Find best path through hidden states given observation states. Backtrace transitions from highest probability final state.
        Probabilities are calcualted given previous hidden state, emission probabilities, and transition probabilities.

        Args:
            decode_observation_states (np.ndarray): list of observation sates to decode

        Returns:
            np.ndarray: most likely sequence of hidden states
        """        
        
        #Rename things to make it easier to read

        observation_states = self.hmm_object.observation_states
        observation_states_dict = self.hmm_object.observation_states_dict
        hidden_states = self.hmm_object.hidden_states
        hidden_states_dict = self.hmm_object.hidden_states_dict
        prior_prob = self.hmm_object.prior_probabilities
        emission_prob = self.hmm_object.emission_probabilities
        transition_prob = self.hmm_object.transition_probabilities

        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), len(hidden_states)))
        
        best_path =np.zeros((len(decode_observation_states), len(hidden_states)), dtype=int)       
        
        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.

        start_obs = observation_states_dict[decode_observation_states[0]]     
        delta = np.multiply(prior_prob, emission_prob[:, start_obs])
        #delta = delta / np.sum(delta) #probabilities must sum to 1, normalizing deltas

        path[0, :] = delta #add initial delta probabilities to track in path

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):
            curr_obs = observation_states_dict[decode_observation_states[trellis_node]]
            prev_delta = path[trellis_node - 1, :] #delta has prior prob and emission prob calculated
            # loop through each hidden state to find probabilties and then find max and index to update best_path and deltas
            for hidden_state in range(len(hidden_states)):
                # probabilities of transitioning from previous state to hidden state using previous delta
                prod_delta_trans = np.multiply(prev_delta, transition_prob[:, hidden_state])

                max_prob_idx = np.argmax(prod_delta_trans) #index of most likely previous state
                max_prob_val = prod_delta_trans[max_prob_idx] #max prob of transitioning between hidden states

                #update prob with observed emission state prob
                prod_delta_trans_emission = np.multiply(max_prob_val, emission_prob[hidden_state, curr_obs])
                #update delta and add to path
                path[trellis_node,hidden_state] = np.max(prod_delta_trans_emission)
                #store index of max prob
                best_path[trellis_node,hidden_state] = max_prob_idx
        
        # get final state by returning index of max value in last row of path/delta matrix (last hidden state)
        final_state = np.argmax(path[-1, :])
        
        #backtrace
        best_hidden_state_path = [final_state]
        for state in range(len(decode_observation_states) - 1, 0, -1):
            prev_state = best_path[state, best_hidden_state_path[-1]]
            best_hidden_state_path.append(prev_state)
        best_hidden_state_path.reverse()
        
        #convert to words
        best_hidden_state_path = np.array([hidden_states_dict[i] for i in best_hidden_state_path])

        return best_hidden_state_path