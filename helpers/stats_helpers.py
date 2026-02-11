import time
import numpy as np
from typing import Dict

# =============================================================================
# CLUSTER RANDOMIZATION
# =============================================================================

def generate_time_blocks(
        T: int, 
        time_block_length: int
        ) -> np.ndarray:
    """Generate array of time block indices for each round."""
    block_id = np.arange(T) // time_block_length
    time_cluster_matrix = np.zeros((T, int(np.ceil(T / time_block_length))))
    time_cluster_matrix[np.arange(T), block_id] = 1
    return time_cluster_matrix

def generate_cluster_treatments(
        cluster_matrix: np.ndarray,
        time_cluster_matrix: np.ndarray,
        num_W: int=1, #Number of treatment assignment matrices to generate
        p_treat: float=0.5
        ) -> np.ndarray:
    
    (n,n_c) = cluster_matrix.shape
    (T,T_c) = time_cluster_matrix.shape
    treatment_probs = p_treat * np.ones((n_c,T_c,num_W))
    cluster_treatment = (np.random.rand(n_c,T_c,num_W) >= treatment_probs).astype(int)
    return np.transpose(np.tensordot(np.tensordot(cluster_matrix, cluster_treatment, axes=([1],[0])), time_cluster_matrix.T, axes=([1],[0])), axes=(0, 2, 1))

# =============================================================================
# PROPENSITY SCORE (EXPOSURE PROB)
# =============================================================================
def exposure_mapping_vanilla(
        treatment_array: np.ndarray,
        adj_matrix: np.ndarray,
        time_adj_matrix: np.ndarray,
        delta: float=0
        ) -> Dict:
    """
    Compute spatio-temporal exposure mapping.
    Input:
        arms_array: N x T binary matrix of treatment assignments (0 or 1)
        adj_map: Adjacency map where adj_map[i] = list of neighbors of node i
        recency: int, temporal window size r (default=1)
    Returns:
        dict containing exposure_1_array, exposure_0_array, dof, and num_spatial_neighbors
    """
    exposure_1_array = treatment_array 
    exposure_0_array = (1 - treatment_array) 
    return {
        "exposure_1": exposure_1_array,
        "exposure_0": exposure_0_array,
        "dof": np.ones(treatment_array.shape),  # for debugging
    }

def exposure_mapping(
        treatment_array: np.ndarray,
        adj_matrix: np.ndarray,
        time_adj_matrix: np.ndarray,
        delta: float=0
        ) -> Dict:
    """
    Compute spatio-temporal exposure mapping.
    Input:
        arms_array: N x T binary matrix of treatment assignments (0 or 1)
        adj_map: Adjacency map where adj_map[i] = list of neighbors of node i
        recency: int, temporal window size r (default=1)
    Returns:
        dict containing exposure_1_array, exposure_0_array, dof, and num_spatial_neighbors
    """
    adj_matrix = adj_matrix - np.eye(adj_matrix.shape[0])
    time_adj_matrix = time_adj_matrix - np.eye(time_adj_matrix.shape[0])
    treated_neighb = np.transpose(np.tensordot(np.tensordot(adj_matrix, treatment_array, axes=([1],[0])), time_adj_matrix.T, axes=([1],[0])), axes=(0, 2, 1))
    total_neighb = np.transpose(np.tensordot(np.tensordot(adj_matrix, np.ones(treatment_array.shape), axes=([1],[0])), time_adj_matrix.T, axes=([1],[0])), axes=(0, 2, 1))
    exposure_1_array = treatment_array * (treated_neighb >= (1-delta)*total_neighb).astype(int)  # 1 if self treated AND (1-delta) fraction of neighbors are treated
    exposure_0_array = (1 - treatment_array) * (treated_neighb <= delta*total_neighb).astype(int)  # 1 if self control AND (1-delta) fraction of neighbors are untreated
    return {
        "exposure_1": exposure_1_array,
        "exposure_0": exposure_0_array,
        "dof": total_neighb,  # for debugging
    }

def exposure_mapping_old(
        treatment_array: np.ndarray,
        adj_matrix: np.ndarray,
        time_adj_matrix: np.ndarray,
        delta: float=0
        ) -> Dict:
    """
    Compute spatio-temporal exposure mapping.
    Input:
        arms_array: N x T binary matrix of treatment assignments (0 or 1)
        adj_map: Adjacency map where adj_map[i] = list of neighbors of node i
        recency: int, temporal window size r (default=1)
    Returns:
        dict containing exposure_1_array, exposure_0_array, dof, and num_spatial_neighbors
    """ 
    treated_neighb = np.transpose(np.tensordot(np.tensordot(adj_matrix, treatment_array, axes=([1],[0])), time_adj_matrix.T, axes=([1],[0])), axes=(0, 2, 1))
    total_neighb = np.transpose(np.tensordot(np.tensordot(adj_matrix, np.ones(treatment_array.shape), axes=([1],[0])), time_adj_matrix.T, axes=([1],[0])), axes=(0, 2, 1))
    exposure_1_array = (treated_neighb >= (1-delta)*total_neighb).astype(int)  # 1 if (1-delta) fraction of neighbors are treated
    exposure_0_array = (treated_neighb <= delta*total_neighb).astype(int)  # 1 if (1-delta) fraction of neighbors are untreated
    return {
        "exposure_1": exposure_1_array,
        "exposure_0": exposure_0_array,
        "dof": total_neighb,  # for debugging
    }

def empirical_propensity_scores(
        treatment_array: np.ndarray,
        adj_matrix: np.ndarray,
        time_adj_matrix: np.ndarray,
        delta: float=0
        ) -> Dict:
    
    exposure_results = exposure_mapping(treatment_array, adj_matrix, time_adj_matrix, delta)
    
    # Return averaged propensity scores
    return {
        'propensity_1': np.average(exposure_results['exposure_1'], axis=2),
        'propensity_0': np.average(exposure_results['exposure_0'], axis=2),
        'dof': exposure_results['dof']  # dof of the last iteration (for debugging)
    }

def compute_theoretical_propensity_scores_bernoulli_unit_randomization(
    adj_matrix: np.ndarray,
    time_adj_matrix: np.ndarray,
    p_treat: float=0.5
) -> tuple:
    """ Compute the THEORETICAL propensity scores P(X_ita^r = 1)    
    Input:
        adj_map: Adjacency map where adj_map[i] = list of neighbors of node i (excluding i itself)
        recency: int, temporal window size r (default=1)
        p1: float, probability of assigning arm 1 to any individual node/time (default=0.5)
    Returns: tuple (propensity_1_array, propensity_0_array), both N x T arrays
    """
    num_nodes = adj_matrix.shape[0]
    num_time = time_adj_matrix.shape[0]
    propensity_1_array, propensity_0_array = np.zeros((num_nodes, num_time)), np.zeros((num_nodes, num_time))
    dof = np.outer(np.sum(adj_matrix,axis=1), np.sum(time_adj_matrix,axis=1)) # degree of freedom = #clusters whose arms matters for (i,t)'s expo map
    propensity_1_array, propensity_0_array = np.power(p_treat,dof), np.power(1-p_treat,dof)
    return (propensity_1_array, propensity_0_array)

# =============================================================================
# HORVITZ-THOMPSON ESTIMATOR
# =============================================================================

def horvitz_thompson(
        rewards_array: np.ndarray,
        exposure_1_array: np.ndarray,
        exposure_0_array: np.ndarray,
        propensity_1_array: np.ndarray,
        propensity_0_array: np.ndarray) -> Dict:

    (n,T,num_sims) = rewards_array.shape

    HT_weights_0 = np.zeros((np.shape(propensity_0_array)))
    np.divide(1, propensity_0_array, out = HT_weights_0, where=propensity_0_array != 0)
    Y_hat_0 = exposure_0_array * rewards_array * HT_weights_0[:,:,np.newaxis]
    ate_estimate_ht_arm0 = np.average(Y_hat_0, axis = (0,1))

    HT_weights_1 = np.zeros((np.shape(propensity_1_array)))
    np.divide(1, propensity_1_array, out = HT_weights_1, where=propensity_1_array != 0)
    Y_hat_1 = exposure_1_array * rewards_array * HT_weights_1[:,:,np.newaxis]
    ate_estimate_ht_arm1 = np.average(Y_hat_1, axis = (0,1))
    
    ate_estimate_ht = ate_estimate_ht_arm1 - ate_estimate_ht_arm0

    return {
        'ate_estimate_ht_arm0': ate_estimate_ht_arm0,
        'ate_estimate_ht_arm1': ate_estimate_ht_arm1,
        'ate_estimate_ht': ate_estimate_ht,
    }


# =============================================================================
# HAJEK ESTIMATOR
# =============================================================================

def hajek(
        rewards_array: np.ndarray,
        exposure_1_array: np.ndarray,
        exposure_0_array: np.ndarray,
        propensity_1_array: np.ndarray,
        propensity_0_array: np.ndarray) -> Dict:

    (n,T,num_sims) = rewards_array.shape

    HT_weights_0 = np.zeros((np.shape(propensity_0_array)))
    np.divide(1, propensity_0_array, out = HT_weights_0, where=propensity_0_array != 0)
    Y_hat_0 = np.sum(exposure_0_array * rewards_array * HT_weights_0[:,:,np.newaxis], axis = (0,1))
    denom_0 = np.sum(exposure_0_array * HT_weights_0[:,:,np.newaxis], axis = (0,1))

    ate_estimate_hajek_arm0 = np.zeros(num_sims)
    np.divide(Y_hat_0, denom_0, out = ate_estimate_hajek_arm0, where=denom_0 != 0)

    HT_weights_1 = np.zeros((np.shape(propensity_1_array)))
    np.divide(1, propensity_1_array, out = HT_weights_1, where=propensity_1_array != 0)
    Y_hat_1 = np.sum(exposure_1_array * rewards_array * HT_weights_1[:,:,np.newaxis], axis = (0,1))
    denom_1 = np.sum(exposure_1_array * HT_weights_1[:,:,np.newaxis], axis = (0,1))

    ate_estimate_hajek_arm1 = np.zeros(num_sims)
    np.divide(Y_hat_1, denom_1, out = ate_estimate_hajek_arm1, where=denom_1 != 0)
    
    ate_estimate_hajek = ate_estimate_hajek_arm1 - ate_estimate_hajek_arm0

    return {
        'ate_estimate_hajek_arm0': ate_estimate_hajek_arm0,
        'ate_estimate_hajek_arm1': ate_estimate_hajek_arm1,
        'ate_estimate_hajek': ate_estimate_hajek,
    }
