#!/usr/bin/env python3
import time, importlib, sys, os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

from helpers import utils, graph_helpers, mdp_helpers, stats_helpers, print_nicely, simulation_setup

def main():

    # # n_max = 1000
    # # coords_array = graph_helpers.generate_random_points(n_max)
    # # max_inventory = 30*np.ones(n_max)
    # # initial_state = (np.random.random(max_inventory.shape) * max_inventory).astype(int)
    # # np.savez('unifom_spatial_network.npz', coords_array=coords_array, max_inventory = max_inventory, initial_state = initial_state)  
    # n = 100
    # kappa = [0.2] # kappa = [0.1, 0.15, 0.2, 0.25, 0.3] 
    # num_cell_per_dim = [4,6,8,10,15]
    # loaded_network = np.load('unifom_spatial_network.npz')
    # coords_array = loaded_network['coords_array'][:n,:]
    # max_inventory = loaded_network['max_inventory'][:n]
    # initial_state = loaded_network['initial_state'][:n]

    # # (coords_array, max_inventory) = simulation_setup.setup_Hotel_Dataset('Hotels.csv', 10)
    # # initial_state = (np.random.random(max_inventory.shape) * max_inventory).astype(int)
    # # np.savez('hotel_network.npz', coords_array=coords_array, max_inventory = max_inventory, initial_state = initial_state)
    kappa = [0.035]
    num_cell_per_dim = [10,20,30,40,50,60,70]
    loaded_network = np.load('hotel_network.npz')
    coords_array = loaded_network['coords_array']
    max_inventory = loaded_network['max_inventory']
    initial_state = loaded_network['initial_state']

    for k in kappa:
        print(f'Kappa: {k}')
        adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, k)
        print(f'Average number of neighbors: {np.mean(np.sum(adj_matrix,axis=1)):.2f}')
        print(f'Std dev number of neighbors: {np.std(np.sum(adj_matrix,axis=1)):.2f}')
        for ncpd in num_cell_per_dim:
            print(f'Num_cells_per_dim: {ncpd}')
            cluster_matrix = graph_helpers.spatial_clustering_map(coords_array, ncpd)
            print(f'Total number of clusters: {cluster_matrix.shape[1]}')
            neighb_clusters = (np.matmul(adj_matrix,cluster_matrix) > 0).astype(int)
            print(f'Average number of neighboring clusters: {np.mean(np.sum(neighb_clusters,axis=1)):.2f}')
            print(f'Std dev number of neighboring clusters: {np.std(np.sum(neighb_clusters,axis=1)):.2f}')

if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
