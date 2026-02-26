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
    if len(sys.argv) > 1:
        network = sys.argv[1] # 'Hotel' or 'Uniform'
    else:
        print("Need to specify network")

    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(OUTPUT_DIR, f"run_network_{network}_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    print(f"Saving data to {run_folder}")
    
    log_path = os.path.join(run_folder, "_network_simulation_log.txt")
    logger = LogToFile(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    if network == 'Uniform':
        # n_max = 1000
        # coords_array = graph_helpers.generate_random_points(n_max)
        # max_inventory = 30*np.ones(n_max)
        # initial_state = (np.random.random(max_inventory.shape) * max_inventory).astype(int)
        # np.savez('unifom_spatial_network.npz', coords_array=coords_array, max_inventory = max_inventory, initial_state = initial_state)  
        n = 100
        kappa = [0.1, 0.15, 0.2, 0.25, 0.3] 
        num_cells_per_dim = [4,6,8,10,15]
        loaded_network = np.load('unifom_spatial_network.npz')
        coords_array = loaded_network['coords_array'][:n,:]
        # max_inventory = loaded_network['max_inventory'][:n]
        # initial_state = loaded_network['initial_state'][:n]
    elif network == 'Hotel':
        # (coords_array, max_inventory) = simulation_setup.setup_Hotel_Dataset('Hotels.csv', 10)
        # initial_state = (np.random.random(max_inventory.shape) * max_inventory).astype(int)
        # np.savez('hotel_network.npz', coords_array=coords_array, max_inventory = max_inventory, initial_state = initial_state)
        kappa = [0.01, 0.02, 0.035, 0.05, 0.07]
        num_cells_per_dim = [10,20,30,40,50,60,70]
        loaded_network = np.load('hotel_network.npz')
        coords_array = loaded_network['coords_array']
        # max_inventory = loaded_network['max_inventory']
        # initial_state = loaded_network['initial_state']
    else:
        print("Error, network must be either Hotel or Uniform dataset")
        return
    
    stats =[]
    for k in kappa:
        print(f'Kappa: {k}')
        adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, k)
        print(f'Average number of neighbors: {np.mean(np.sum(adj_matrix,axis=1)):.2f}')
        print(f'Std dev number of neighbors: {np.std(np.sum(adj_matrix,axis=1)):.2f}')
        for ncpd in num_cells_per_dim:
            print(f'Num_cells_per_dim: {ncpd}')
            cluster_matrix = graph_helpers.spatial_clustering_map(coords_array, ncpd)
            print(f'Total number of clusters: {cluster_matrix.shape[1]}')
            neighb_clusters = (np.matmul(adj_matrix,cluster_matrix) > 0).astype(np.int8)
            print(f'Average number of neighboring clusters: {np.mean(np.sum(neighb_clusters,axis=1)):.2f}')
            print(f'Std dev number of neighboring clusters: {np.std(np.sum(neighb_clusters,axis=1)):.2f}')
            stats.append([k,ncpd,np.mean(np.sum(adj_matrix,axis=1)),np.std(np.sum(adj_matrix,axis=1)),cluster_matrix.shape[1],np.mean(np.sum(neighb_clusters,axis=1)),np.std(np.sum(neighb_clusters,axis=1))])

    network_stats = pd.DataFrame(stats, columns=['kappa','num_cells_per_dim','avg_neighb','std_dev_neighb','num_clusters','avg_neighb_clusters','std_dev_neighb_clusters'])
    network_stats.to_csv(run_folder+f'/{network}_network_stats.csv')

    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()

if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
