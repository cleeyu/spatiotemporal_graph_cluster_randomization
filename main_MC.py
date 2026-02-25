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
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Set up logging to temporary file first (will move to correct folder later)
    temp_log_path = os.path.join("results", "temp_MC_simulation_log.txt")
    logger = LogToFile(temp_log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    # # n_max = 1000
    # # coords_array = graph_helpers.generate_random_points(n_max)
    # # max_inventory = 30*np.ones(n_max)
    # # initial_state = (np.random.random(max_inventory.shape) * max_inventory).astype(int)
    # # np.savez('unifom_spatial_network.npz', coords_array=coords_array, max_inventory = max_inventory, initial_state = initial_state)  
    # n = 100
    # kappa = [0.2] # kappa = [0.1, 0.15, 0.2, 0.25, 0.3] 
    # loaded_network = np.load('unifom_spatial_network.npz')
    # coords_array = loaded_network['coords_array'][:n,:]
    # max_inventory = loaded_network['max_inventory'][:n]
    # initial_state = loaded_network['initial_state'][:n]

    # # (coords_array, max_inventory) = simulation_setup.setup_Hotel_Dataset('Hotels.csv', 10)
    # # initial_state = (np.random.random(max_inventory.shape) * max_inventory).astype(int)
    # # np.savez('hotel_network.npz', coords_array=coords_array, max_inventory = max_inventory, initial_state = initial_state)
    kappa = 0.035
    loaded_network = np.load('hotel_network.npz')
    coords_array = loaded_network['coords_array']
    max_inventory = loaded_network['max_inventory']
    initial_state = loaded_network['initial_state']

    num_rounds = 10**5
    num_sims_apx_GATE = 10**4
    
    adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, kappa)
    MC_model = simulation_setup.setup_MC_model(adj_matrix, num_rounds, max_inventory, initial_state)
    results_GATE = MC_model.estimate_GATE(num_sims_apx_GATE)
    true_GATE = results_GATE[0]
    std_dev_GATE = results_GATE[3]
    diff_means = results_GATE[4]

    print(f"GATE Estimate: {true_GATE:.4f}")
    print(f"Std Dev: {std_dev_GATE:.4f}")

    for t in range(10):
        truncate = int((t+1)*0.1*num_rounds)
        print(f"GATE Estimate up to T={truncate}: {np.mean(diff_means[:truncate]):.4f}")

    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join("results", f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)

    # Move log file to the correct results folder
    final_log_path = os.path.join(run_folder, "_MC_simulation_log.txt")
    import shutil
    shutil.move(temp_log_path, final_log_path)

if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
