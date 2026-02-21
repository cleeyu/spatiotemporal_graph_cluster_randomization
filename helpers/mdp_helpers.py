import numpy as np
from typing import Tuple, Dict
from . import utils

class InventoryMarkovChain:

    def __init__(self, 
            max_inventory: np.ndarray, # max_inventory for each unit, size is n
            adj_matrix: np.ndarray,  # size is n x n
            num_rounds: int, 
            C_baseline: np.ndarray, # size is n x T
            C_slope: np.ndarray, # size is n x T
            C_lazy: np.ndarray, # size is n x T
            C_alpha: np.ndarray, # size is n x T
            C_beta: np.ndarray, # size is n x T
            C_gamma: np.ndarray, # size is n x T
            C_depart: np.ndarray, # size is n x T
            ):
        """
        Initialize the Markov Chain.
        
        Args:
            states: List of possible states
            transition_matrix: numpy array of shape (n_states, n_states) with P[i,j] = P(s_j|s_i)
            transition_dict: Alternative: Dict mapping (state_i, state_j) to probability
        """
        self.max_inventory = np.maximum(1, max_inventory)
        self.adj_matrix = adj_matrix
        num_nodes = adj_matrix.shape[0]
        self.num_nodes = num_nodes
        self.num_rounds = num_rounds
        
        self.laziness_array = C_lazy
        self.alpha_array = C_alpha
        self.beta_array = C_beta
        self.gamma_array = C_gamma
        self.depart_array = C_depart

        self.B = C_baseline
        self.R = np.divide(C_slope, self.max_inventory[:,np.newaxis])

        self.current_state = None
    
    def compute_move_up(self, treatment_tensor: np.ndarray, use_sigmoid: bool=True) -> np.ndarray:
        """
        Set the treatment tensor for the Markov Chain.
        """
        if (treatment_tensor.shape[0], treatment_tensor.shape[1]) == (self.num_nodes, self.num_rounds):
            self.treatment_tensor = treatment_tensor
        else:
            self.treatment_tensor = np.zeros((self.num_nodes, self.num_rounds,1))

        denom = np.tensordot(self.adj_matrix, np.ones(treatment_tensor.shape), axes=([1],[0])) - 1
        competition = np.zeros(treatment_tensor.shape)
        np.divide((np.tensordot(self.adj_matrix, self.treatment_tensor, axes=([1],[0])) - self.treatment_tensor), denom, out=competition, where=denom!=0)
        competition = np.minimum(competition,0.75)
        competition = np.maximum(competition,0.25)
        P_it = self.alpha_array[:,:,np.newaxis] + self.beta_array[:,:,np.newaxis] * self.treatment_tensor - self.gamma_array[:,:,np.newaxis] * competition
        if use_sigmoid:
            P_it = 1 / (1 + np.exp(-P_it))
        return P_it / (P_it + self.depart_array[:,:,np.newaxis])
    
    def compute_reward_trajectory(self, state_trajectory) -> np.ndarray:
        """ Compute reward for a given state at a given time """
        return self.B[:,:,np.newaxis] + self.R[:,:,np.newaxis] * state_trajectory
    
    def simulate_MC(self, state_vector: np.ndarray, treatment_tensor: np.ndarray, use_sigmoid: bool=True) -> Dict:
        P_it = self.compute_move_up(treatment_tensor, use_sigmoid)
        num_sims = P_it.shape[2]

        if state_vector.shape != (self.num_nodes,):
            print("Invalid state vector shape ",str(state_vector.shape),", should be (num_nodes,) starting all states at zero")
            state_vector = np.zeros((self.num_nodes))
        
        state_vector = np.outer(state_vector, np.ones((num_sims)))

        not_lazy_mask = (np.random.rand(self.num_nodes, self.num_rounds, num_sims) >= self.laziness_array[:,:,np.newaxis]).astype(int)
        move_up_mask = (np.random.rand(self.num_nodes, self.num_rounds, num_sims) <= P_it).astype(int)
        transition = not_lazy_mask * move_up_mask - not_lazy_mask * (1 - move_up_mask)
        
        state_trajectory = np.zeros((self.num_nodes, self.num_rounds, num_sims))
        for current_time in range(self.num_rounds):
            state_vector = np.maximum(np.minimum(state_vector + transition[:, current_time, :], self.max_inventory[:,np.newaxis]),0)
            state_trajectory[:, current_time, :] = state_vector

        rewards = self.compute_reward_trajectory(state_trajectory)
        return {"state_trajectory": state_trajectory, "rewards":rewards}
