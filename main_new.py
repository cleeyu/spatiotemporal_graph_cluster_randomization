#!/usr/bin/env python3
""" Simulations for Clustered Switchback Experiments. Converted from JuNo to Py script """
import time, importlib, sys, os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# ================================================================================
# CONFIG. PARAMETERS - Modify these to tune the simulation
# ================================================================================

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


# ================================================================================
# MAIN SIMULATION CODE
# ================================================================================

def main():
    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    
    log_path = os.path.join(run_folder, "_simulation_log.txt")
    logger = LogToFile(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    # n = 100
    # kappa = 0.2 # kappa = [0.1, 0.15, 0.2, 0.25, 0.3]
    # num_cells_per_dim = [4,6,8,10,15]
    # loaded_network = np.load('unifom_spatial_network.npz')
    # coords_array = loaded_network['coords_array'][:n,:]
    # max_inventory = loaded_network['max_inventory'][:n]
    # initial_state = loaded_network['initial_state'][:n]
    # MC_param_file = 'Uniform_MC_parameters.npz'
    # print(f"Uniform Spatial Network with {n} nodes and kappa = {kappa}")

    kappa = 0.035
    num_cells_per_dim = [10,20,30,40,50,60,70]
    loaded_network = np.load('hotel_network.npz')
    coords_array = loaded_network['coords_array']
    max_inventory = loaded_network['max_inventory']
    initial_state = loaded_network['initial_state']
    MC_param_file = 'Hotel_MC_parameters.npz'
    print(f"Hotel Network with kappa = {kappa}")

    sim_config = {
        'maximum_T': 10000,
        'num_cells_per_dim': num_cells_per_dim,
        'time_block_length': [20,40,60,80,100], #[3,6,9],
        'recency': [10], 
        'delta': [0.2], #[0,0.1,0.2,0.3],
        'burn_in': [5], #[0,2,4,6],
        'num_monte_carlo_gate': 10**3,
        'num_iter_est': 10**4,
        'coords_array': coords_array,
        'kappa': kappa,
        'max_inventory': max_inventory,
        'initial_state': initial_state,
    }
    all_results = simulation_setup.run_simulation(sim_config,run_folder+'/all_data.pkl',run_folder+'/all_results.csv', load_MC_param_file = MC_param_file)

    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()
    
    print(f"Simulation completed. Log saved to: {log_path}")

if __name__ == "__main__":
    """Run the simulation when script is executed directly"""
    print("Starting Clustered Switchback Experiment Simulation...")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        main()
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise
