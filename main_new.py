#!/usr/bin/env python3
""" Simulations for Clustered Switchback Experiments. Converted from JuNo to Py script """
import time, importlib, sys, os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

# ================================================================================
# CONFIG. PARAMETERS - Modify these to tune the simulation
# ================================================================================

# parameters: Network and spatial clustering
N = 1                      # Number of nodes (must be perfect square if generating lattice graph)
NUM_CELLS_PER_DIM = 1              # Width of squares for cluster randomization
KAPPA = 0.1                      # Kappa parameter for interference graph generation

# parameters: Design and estimator 
T = 100                      # Number of time periods
TIME_BLOCK_LENGTH = 50         # Length of time blocks for cluster randomization
NUM_STATES = 30                 # Number of states in the MDP

# parameters: Simulation
INITIAL_STATE = 15              # Initial state for MDP simulation
NUM_MONTE_CARLO_ATE = 1000       # Number of simulations for Monte Carlo ATE approximation
NUM_PROP_SCORE_SIMS = 1000      # Number of simulations for propensity score computation
NUM_ITER_EST = 10000               # Number of iterations for main HT/Hajek estimator simulation

# Print and save
OUTPUT_DIR = "results"         # Directory to save results
OUTPUT_FILENAME = "sim_results.csv"  # CSV filename for results

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
    sim_config = {
        "n": N,
        "kappa": KAPPA,
        "T": T,
        "num_states": NUM_STATES,
        "initial_state": INITIAL_STATE,
        "num_cells_per_dim":  NUM_CELLS_PER_DIM,
        "time_block_length": TIME_BLOCK_LENGTH,
    }
    simulation_setup.run_simulation(sim_config,NUM_MONTE_CARLO_ATE,NUM_ITER_EST)

if __name__ == "__main__":
    """Run the simulation when script is executed directly"""
    print("Starting Clustered Switchback Experiment Simulation...")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        main()
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise
