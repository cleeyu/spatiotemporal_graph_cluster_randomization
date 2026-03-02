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
        expt_type = sys.argv[2] # e.g. 'switchback'
    else:
        print("Need to specify network")

    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(OUTPUT_DIR, f"run_est_{network}_{expt_type}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving data to {run_folder}")
    
    log_path = os.path.join(run_folder, "_simulation_log.txt")
    logger = LogToFile(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    if network == 'Uniform':
        n = 100
        kappa = 0.2 # kappa = [0.1, 0.15, 0.2, 0.25, 0.3] 
        num_cells_per_dim = [1,2,3,4,5,6,8,10,15]
        # 1/(2*kappa) = 2.5
        num_cells_per_dim = [25,30]
        loaded_network = np.load('unifom_spatial_network.npz')
        coords_array = loaded_network['coords_array'][:n,:]
        print(f"Uniform Spatial Network with {n} nodes and kappa = {kappa}")
        GATE_est_file = 'Uniform_true_GATE.csv'
    elif network == 'Hotel':
        kappa = 0.035
        num_cells_per_dim = [5,10,12,14,16,18,20,30]
        # 1/(2*kappa) = 14
        num_cells_per_dim = [10,20,30,40,50,60,70]
        loaded_network = np.load('hotel_network.npz')
        coords_array = loaded_network['coords_array']
        print(f"Hotel Network with kappa = {kappa}")
        GATE_est_file = 'Hotel_true_GATE.csv'
    else:
        print("Error, network must be either Hotel or Uniform dataset")
        return

    adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, kappa)
    
    start_time = time.time()

    print("\nLoad Data and Compute estimates")
    GATE_est = pd.read_csv(GATE_est_file)

    delta = [0.2] # [0,0.1,0.2,0.3]
    time_block_length = [20,40,60,80,100,200,400,600,800,1000]
    
    parameter_type = 'fraction'
    if parameter_type == 'fraction':
        recency = np.ceil(tbl*np.array([0.25,0.5,1,1.5,2]))
        burn_in = np.ceil(tbl*np.array([0,0.1,0.3,0.5]))
    else:
        recency = [5,10,15,20,40,80,120,160,200]
        burn_in = [0,5,10,15,20,40,60,80,100,200,400,600,800,1000,2000]

    if expt_type == 'switchback':
        pairs = [(ncpd,tbl) for ncpd in [1] for tbl in time_block_length]
    elif expt_type == 'clusterRD':
        pairs = [(ncpd,tbl) for ncpd in num_cells_per_dim for tbl in [10000]]
    if expt_type == 'spatiotemporal':
        pairs = [(ncpd,tbl) for ncpd in num_cells_per_dim for tbl in time_block_length]

    for ncpd,tbl in pairs:
        print(f'Load data for num_cells_per_dim = {ncpd}, time_block_length = {tbl}')

        data = (ncpd,tbl,np.load(f'results/run_expt_{network}_{expt_type}/arms_array_{ncpd}_{tbl}.npy').astype(np.uint8),np.load(f'results/run_expt_{network}_{expt_type}_new/rewards_{ncpd}_{tbl}.npy').astype(np.float32))

        print('Data loaded')
        total_runtime_seconds = time.time() - start_time
        print(f"Elapsed time: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")

        all_results = pd.DataFrame()

        # Compute estimates
        print("\nCompute HT/Hajek estimates")
        parameter_data_combo = [(r,d) for r in recency for d in delta]
        HT_Hajek_results = Parallel(n_jobs=5)(
            delayed(simulation_setup.compute_HT_Hajek_estimates)(r,d,data,adj_matrix) for r,d in parameter_data_combo
        )
        all_results = pd.concat(HT_Hajek_results, join='outer', ignore_index=True)

        print("\nCompute Diff-Means estimates")
        DM_results = simulation_setup.compute_DM_estimates(burn_in,data)
        all_results = pd.concat([all_results,DM_results],axis=0, join='outer', ignore_index=True)
        
        if expt_type == 'spatiotemporal':
            all_results['frac_param'] = 'yes'
            
        all_results['kappa'] = kappa
        all_results = pd.merge(all_results, GATE_est, on='T', how='left')

        simulation_setup.print_logs(all_results)
        save_estimates_f = run_folder+f'/{network}_estimates_{ncpd}_{tbl}.csv'
        all_results.to_csv(save_estimates_f)
        
        del data
        del all_results
        gc.collect()             
        
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