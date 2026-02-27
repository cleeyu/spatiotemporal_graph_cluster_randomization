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
    if len(sys.argv) > 1:
        network = sys.argv[1] # 'Hotel' or 'Uniform'
    else:
        print("Need to specify network")

    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(OUTPUT_DIR, f"run_expt_{network}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving data to {run_folder}")
    
    log_path = os.path.join(run_folder, "_simulation_log.txt")
    logger = LogToFile(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    if network == 'Uniform':
        n = 100
        kappa = 0.2 # kappa = [0.1, 0.15, 0.2, 0.25, 0.3]
        # num_cells_per_dim = [1] # [4,6,8,10,15]
        num_cells_per_dim = [2,4,6,8,10,15,20,25,30]
        loaded_network = np.load('unifom_spatial_network.npz')
        coords_array = loaded_network['coords_array'][:n,:]
        max_inventory = loaded_network['max_inventory'][:n]
        initial_state = loaded_network['initial_state'][:n]
        MC_param_file = 'Uniform_MC_parameters.npz'
        print(f"Uniform Spatial Network with {n} nodes and kappa = {kappa}")
    elif network == 'Hotel':
        kappa = 0.035
        # num_cells_per_dim = [1] # [10,20,30,40,50,60,70]
        num_cells_per_dim = [5,10,20,30,40,50,60,70,80,90]
        loaded_network = np.load('hotel_network.npz')
        coords_array = loaded_network['coords_array']
        max_inventory = loaded_network['max_inventory']
        initial_state = loaded_network['initial_state']
        MC_param_file = 'Hotel_MC_parameters.npz'
        print(f"Hotel Network with kappa = {kappa}")
    else:
        print("Error, network must be either Hotel or Uniform dataset")
        return

    if len(sys.argv) > 2:
        num_cells_per_dim = [int(sys.argv[2])]

    num_rounds = 10**4
    time_block_length = [num_rounds]
    # time_block_length = [20,40,60,80,100,200,400,600,800,1000,2000,4000,6000,8000,10000]
    num_iter_est = 10**3
    print(f'num_rounds = {num_rounds}, num_iter_est = {num_iter_est}')

    n = coords_array.shape[0]

    start_time = time.time()
    print("\nSetup Interference Graph and Markov Chain model\n")

    # Create interference graph
    adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, kappa)

    # Setup MC model
    MC_model = simulation_setup.load_MC_model(adj_matrix, num_rounds, max_inventory, initial_state, MC_param_file)

    # Sample design and run simulation
    print(f"\nGenerating treatment vectors and running estimator simulation")

    # pairs = [(ncpd,tbl) for ncpd in num_cells_per_dim for tbl in time_block_length]
    # Parallel(n_jobs=-1)(
    #     delayed(simulation_setup.simulate_experiment)(ncpd,tbl,coords_array,num_rounds,MC_model,num_iter_est,save_data_folder = run_folder) for ncpd,tbl in pairs
    # )
    for ncpd in num_cells_per_dim:
        for tbl in time_block_length:
            print(f'num_cells_per_dim = {ncpd}, time_block_length = {tbl}')
            simulation_setup.simulate_experiment(ncpd,tbl,coords_array,num_rounds,MC_model,num_iter_est,save_data_folder = run_folder)
            total_runtime_seconds = time.time() - start_time
            print(f"Elapsed time: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")

    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()
    

if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise