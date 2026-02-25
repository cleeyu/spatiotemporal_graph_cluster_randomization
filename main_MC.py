#!/usr/bin/env python3
import time, importlib, sys, os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt


# Print and save
OUTPUT_DIR = "results"         # Directory to save results

# ================================================================================
# Class for saving
# ================================================================================

class LogToFile:
    """Simple class to capture stdout to both console and log file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# Set up dir
current_dir = os.getcwd()
print("current dir:", current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from helpers import utils, graph_helpers, mdp_helpers, stats_helpers, print_nicely, simulation_setup

def main():
    network = 'Hotel' # 'Hotel' or 'Uniform'

    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(OUTPUT_DIR, f"run_MC_{network}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving data to {run_folder}")
    
    log_path = os.path.join(run_folder, "_MC_simulation_log.txt")
    logger = LogToFile(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    if network == 'Uniform':
        n = 100
        kappa = 0.2 # kappa = [0.1, 0.15, 0.2, 0.25, 0.3] 
        loaded_network = np.load('unifom_spatial_network.npz')
        coords_array = loaded_network['coords_array'][:n,:]
        max_inventory = loaded_network['max_inventory'][:n]
        initial_state = loaded_network['initial_state'][:n]
        print(f"Uniform Spatial Network with {n} nodes and kappa = {kappa}")
        MC_param_file = 'Uniform_MC_parameters.npz'
        GATE_est_file = 'Uniform_true_GATE.csv'
    elif network == 'Hotel':
        kappa = 0.035
        loaded_network = np.load('hotel_network.npz')
        coords_array = loaded_network['coords_array']
        max_inventory = loaded_network['max_inventory']
        initial_state = loaded_network['initial_state']
        print(f"Hotel Network with kappa = {kappa}")
        MC_param_file = 'Hotel_MC_parameters.npz'
        GATE_est_file = 'Hotel_true_GATE.csv'
    else:
        print("Error, network must be either Hotel or Uniform dataset")
        return

    num_rounds = 10**4
    num_sims_apx_GATE = 10**3
    
    adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, kappa)
    # MC_model = simulation_setup.setup_MC_model(adj_matrix, num_rounds, max_inventory, initial_state, save_f = MC_param_file)
    MC_model = simulation_setup.load_MC_model(adj_matrix, num_rounds, max_inventory, initial_state,MC_param_file)
    results_GATE = MC_model.estimate_GATE(num_sims_apx_GATE)
    true_GATE = results_GATE[0]
    std_dev_GATE = results_GATE[3]
    diff_means = results_GATE[4]

    print(f"GATE Estimate: {true_GATE:.4f}")
    print(f"Std Dev: {std_dev_GATE:.4f}")

    rounds = []
    est_means = []
    for t in range(10):
        truncate = int((t+1)*0.1*num_rounds)
        rounds.append(truncate)
        est_means.append(np.mean(diff_means[:truncate]))
        print(f"GATE Estimate up to T={truncate}: {est_means[-1]:.4f}")

    GATE_est_results = pd.DataFrame()
    GATE_est_results['T'] = rounds
    GATE_est_results['true_GATE'] = est_means
    GATE_est_results.to_csv(GATE_est_file, index=False)

    np.save(run_folder+'/diff_means.npy', diff_means)

    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()
    
if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
