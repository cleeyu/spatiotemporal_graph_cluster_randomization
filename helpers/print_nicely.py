import numpy as np
from typing import Tuple, Dict, Any
import time

def print_progress_ht(
        iter: int, 
        num_iter_simulation: int, 
        start_time: float,
        est_result: dict,
        rewards_array: np.ndarray,
        ht_result: np.ndarray,
        true_ATE: float,
        print_freq: int = 10
        ) -> None:
    """
    Print progress information for iterative processes with HT estimation details.
    Input:
        iter (int): Current iteration number (0-indexed)
        num_iter_simulation (int): Total number of iterations
        start_time (float): Start time from time.time()
        est_result (dict): Dictionary containing estimation results with keys:
            - 'exposure_0', 'exposure_1': Exposure map for treatment 0,1
            - 'ate_estimate_ht': HT ATE estimate for current iteration
        rewards_array (np.ndarray): Array of reward values
        ht_result (np.ndarray): Array of HT estimates from all runs so far
        true_ATE (float): True average treatment effect for bias calculation
        print_freq (int): How often to print progress (default: 10)
    """
    if (iter+1) % print_freq != 0 and iter+1 != num_iter_simulation:
        return
    print("-"*20)
    elapsed = time.time() - start_time
    avg_time_per_iter = elapsed / (iter+1)
    eta = avg_time_per_iter * (num_iter_simulation - iter - 1)
    print(f"Iteration {iter+1:3d}/{num_iter_simulation} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

    print(f"sum of expo_map (0,1 resp): {est_result['exposure_0'].sum():.2f}, {est_result['exposure_1'].sum():.2f}")
    supp_expo_map_1, supp_expo_map_0 = np.nonzero(est_result['exposure_1']), np.nonzero(est_result['exposure_0'])
    print(f"HT est in this iter: {est_result['ate_estimate_ht']:.2f}")
    print(f"avg reward at supp(expo_0), supp(expo_1): {rewards_array[supp_expo_map_0].mean():.2f}, {rewards_array[supp_expo_map_1].mean():.2f}")
    
    # HT statistics averaged over all runs so far
    mean_HT_est, var_HT_est= ht_result[:iter+1].mean(), ht_result[:iter+1].var() 
    print("\n" + f"mean_HT_est: {mean_HT_est:.4f}, bias: {mean_HT_est - true_ATE:.4f}")
    print(f"var_HT_est: {var_HT_est:.6f}, std_HT_est: {np.sqrt(var_HT_est):.4f}")


def print_dict(
        data: Dict[str, Any], 
        title: str = "Dictionary",
        precision: int = 4
        ) -> None:
    """ Print a single-layer dictionary in a nicely formatted way """
    print(f"\n {title} \n" + "="*len(title))
    if not data:
        print("  (empty)")
        return
    max_key_length = max(len(str(key)) for key in data.keys())  # Calculate the maximum key length for alignment
    
    for key, value in data.items():
        # Format the value based on its type
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                formatted_value = f"{value.item():.{precision}f}" if np.issubdtype(value.dtype, np.floating) else str(value.item())
            else:
                formatted_value = f"array(shape={value.shape}, dtype={value.dtype})"
        elif isinstance(value, (list, tuple)) and len(value) <= 5:
            formatted_value = str(value) # Show small lists/tuples inline
        elif isinstance(value, (list, tuple)):
            formatted_value = f"{type(value).__name__}(length={len(value)})" # Show summary for large lists/tuples
        else:
            formatted_value = str(value)
        print(f"  {key:<{max_key_length}} : {formatted_value}") # Print with aligned formatting
