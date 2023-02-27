"""
UCSF BMI203: Biocomputing Algorithms
Author: tracy lou  
Date: 2/24/2023
Program: biophysics
Description: unit tests for viterbi algorithm
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """
    use an HMM to predict whether R01/R21 funding affects whether a grad student is committed or ambivlent to a rotation lab
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('../data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """
    use an HMM to check whether someone will be on time or late based on traffic
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    #emission and transition prob matrics have same dimension
    assert np.shape(use_case_one_viterbi.hmm_object.emission_probabilities) == np.shape(use_case_one_viterbi.hmm_object.transition_probabilities)
    #length of hidden state sequence should equal length of observed sequence
    decode_observation_states = use_case_one_data['observation_states']
    test_seq = use_case_one_viterbi.best_hidden_state_sequence(decode_observation_states)
    assert len(decode_observation_states) == len(test_seq)

    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_two():
    """use and HMM to predict whether it was sunny or rainy based on a person's mood (happy or sad) with hypothesis that if it's sunny, 
    more likely to be happy
    """

    prior_prob = np.array([0.6, 0.4])
    transition_prob = np.array([[0.7, 0.3], 
                                [0.4, 0.6]])
    emission_prob = np.array([[0.9, 0.1],
                              [0.2, 0.8]])
    
    hidden_states = ['sunny', 'rainy']
    observation_states = ['happy', 'sad']
    
    decode_observation_states = ['happy', 'sad']
    real_seq = np.array(['rainy', 'sunny'])

    hmm_weather = HiddenMarkovModel(observation_states = observation_states,
                                    hidden_states = hidden_states,
                                    prior_probabilities = prior_prob,
                                    transition_probabilities = transition_prob,
                                    emission_probabilities = emission_prob)
    
    viterbi_weather = ViterbiAlgorithm(hmm_object=hmm_weather)

    test_seq = viterbi_weather.best_hidden_state_sequence(decode_observation_states)

    assert np.alltrue(real_seq == test_seq)


def test_user_case_three():
    """use and HMM to predict whether it someone slept enough based on on a person's energy levels
    """
    prior_prob = np.array([0.8, 0.2])
    transition_prob = np.array([[0.6, 0.4], 
                                [0.3, 0.7]])
    emission_prob = np.array([[0.9, 0.1],
                              [0.2, 0.8]])
    
    hidden_states = ['normal sleep', 'sleep deprived']
    observation_states = ['energetic', 'tired']
    
    decode_observation_states = ['energetic', 'energetic', 'tired','tired','tired']
    real_seq = np.array(['normal sleep', 'normal sleep', 'sleep deprived', 'sleep deprived', 'sleep deprived'])

    hmm_sleep = HiddenMarkovModel(observation_states = observation_states,
                                    hidden_states = hidden_states,
                                    prior_probabilities = prior_prob,
                                    transition_probabilities = transition_prob,
                                    emission_probabilities = emission_prob)
    
    viterbi_sleep = ViterbiAlgorithm(hmm_object=hmm_sleep)

    test_seq = viterbi_sleep.best_hidden_state_sequence(decode_observation_states)

    assert np.alltrue(real_seq == test_seq)
