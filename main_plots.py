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
import gc

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
        data_folder_time = sys.argv[2] # e.g. '20260225_090411'
        num_cells_per_dim = [int(sys.argv[3])]
    else:
        print("Need to specify network")

    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(OUTPUT_DIR, f"run_plot_{network}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving data to {run_folder}")
    
    log_path = os.path.join(run_folder, "_simulation_log.txt")
    logger = LogToFile(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    start_time = time.time()

    print("\nLoad Data and Compute estimates")

    time_block_length = [20,40,60,80,100]

    all_stats = pd.DataFrame()
    
    for ncpd in num_cells_per_dim:
        for tbl in time_block_length:
            print(f'Load estimates for num_cells_per_dim = {ncpd}, time_block_length = {tbl}')

            save_estimates_f = f'run_est_{network}_test_{data_folder_time}/{network}_estimates_{ncpd}_{tbl}.csv'
            all_results = pd.read_csv(save_estimates_f)
            
            aggregate_stats = all_results.groupby(['name','num_cells_per_dim', 'time_block_length']).agg({'gate_estimate': ['mean', 'std'], 'true_GATE': ['mean']})
            aggregate_stats[('gate_estimate','bias')] = aggregate_stats[('gate_estimate','mean')] - aggregate_stats[('true_GATE','mean')]

            pd.concat([all_stats, aggregate_stats], ignore_index=True)

            total_runtime_seconds = time.time() - start_time
            print(f"Elapsed time: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")

    all_stats.to_csv(run_folder+'/all_stats.csv')

    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()

    # graph and clustering
    # kappa (fixed)
    # num_cells_per_dim

    # Time horizon and time blocks
    # T (fixed)
    # time_block_length

    # HT/Hajek estimator parameters
    # recency = [10,20,40,80] # [10] # 
    # delta = [0,0.1,0.2,0.3] # [0.2] # [0.1,0.2,0.3] #,

    # DM estimator parameters
    # burn_in = [5,10,15] # [2,4,6,8,10,12,14,16,18]

if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise