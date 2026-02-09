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
N = 30**2                         # Number of nodes (must be perfect square if generating lattice graph)
NUM_CELLS_PER_DIM = 6              # Width of squares for cluster randomization
KAPPA = 3                      # Kappa parameter for interference graph generation

# parameters: Design and estimator 
T = 900                        # Number of time periods
TIME_BLOCK_LENGTH = 5          # Length of time blocks for cluster randomization
NUM_STATES = 5                 # Number of states in the MDP
DELTA = 0.2                   # Delta parameter for exposure computation
RECENCY = 4                    # Recency parameter for exposure computation

# parameters: MDP 
C_LAZY = 0.1                   # Laziness parameter
C_ALPHA = -5                   # Alpha parameter for transition probability
C_BETA = 10                    # Beta parameter for transition probability  
C_GAMMA = 0                    # Gamma parameter for transition probability
C_BASELINE = 100               # Baseline reward parameter
C_SLOPE = 10                   # Slope parameter for reward function

# parameters: Simulation
INITIAL_STATE = 1              # Initial state for MDP simulation
NUM_MONTE_CARLO_ATE = 100       # Number of simulations for Monte Carlo ATE approximation
NUM_PROP_SCORE_SIMS = 10**3      # Number of simulations for propensity score computation
NUM_ITER_EST = 10**3               # Number of iterations for main HT/Hajek estimator simulation

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

from helpers import utils, graph_helpers, mdp_helpers, stats_helpers, print_nicely


# ================================================================================
# MAIN SIMULATION CODE
# ================================================================================

def main():
    """ Main simulation """
    simulation_start_time = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist
    
    # Set up logging to temporary file first (will move to correct folder later)
    temp_log_path = os.path.join(OUTPUT_DIR, "temp_simulation_log.txt")
    logger = LogToFile(temp_log_path)
    original_stdout = sys.stdout
    sys.stdout = logger
    
    print("="*80 + "\nCLUSTERED SWITCHBACK SIMULATION\n" + "="*80)
    print("Setting up simulation logging...")
    
    # Store configuration for output with human-readable names
    config_data = {
        'Network_Size_n': N,
        'Time_Periods_T': T,
        'MDP_States': NUM_STATES,
        'Num_Clusters_Per_Dim': NUM_CELLS_PER_DIM,
        'Time_Block_Length': TIME_BLOCK_LENGTH,
        'Delta_Parameter': DELTA,
        'Recency_Parameter': RECENCY,
        'Interference_Kappa': KAPPA,
        'MDP_Laziness_C_lazy': C_LAZY,
        'Transition_Alpha_C_alpha': C_ALPHA,
        'Transition_Beta_C_beta': C_BETA,
        'Transition_Gamma_C_gamma': C_GAMMA,
        'Reward_Baseline_C_baseline': C_BASELINE,
        'Reward_Slope_C_slope': C_SLOPE,
        'Initial_MDP_State': INITIAL_STATE,
        'Monte_Carlo_ATE_Iterations': NUM_MONTE_CARLO_ATE,
        'Propensity_Score_Simulations': NUM_PROP_SCORE_SIMS,
        'HT_Estimator_Iterations': NUM_ITER_EST
    }
    
    # Step 1: Define instance (interference graph, reward func, transition probability)
    print("\nStep 1: Defining simulation instance...")
    sim_config = {
        "n": N,
        "T": T,
        "num_states": NUM_STATES,
        "num_cells_per_dim":  NUM_CELLS_PER_DIM,
        "time_block_length": TIME_BLOCK_LENGTH,
        "delta": DELTA,
        "recency": RECENCY,
        "kappa": KAPPA
    }
    
    # Create interference graph
    adj_matrix = graph_helpers.generate_interference_graph_from_lattice(
        sqrt_n=int(np.sqrt(sim_config['n'])), 
        kappa=sim_config['kappa']
    )

    cluster_matrix = graph_helpers.generate_clusters_from_lattice(sqrt_n=int(np.sqrt(sim_config['n'])), num_cells_per_dim=sim_config['num_cells_per_dim'])

    time_cluster_matrix = stats_helpers.generate_time_blocks(T=sim_config['T'], time_block_length=sim_config['time_block_length'])
    time_adj_matrix = np.tril(np.ones((sim_config['T'],sim_config['T'])), k=0) - np.tril(np.ones((sim_config['T'],sim_config['T'])), k=-(sim_config['recency'] + 1))
    
    # Generate the Markov Chain 
    MC = mdp_helpers.InventoryMarkovChain(
        max_inventory=sim_config['num_states'],
        adj_matrix=adj_matrix,
        num_rounds=sim_config['T'],
        C_lazy=C_LAZY,
        C_alpha=C_ALPHA,
        C_beta=C_BETA,
        C_gamma=C_GAMMA,
        C_baseline=C_BASELINE,
        C_slope=C_SLOPE
    )

    # Step 2: Find true ATE using Monte Carlo
    print("\nStep 2: Computing true ATE using Monte Carlo...")
    start_time = time.time()

    sim_results_0 = MC.simulate_MC(INITIAL_STATE * np.ones((sim_config['n'])), np.zeros((sim_config['n'], sim_config['T'],NUM_MONTE_CARLO_ATE)), use_sigmoid=True)
    sim_results_1 = MC.simulate_MC(INITIAL_STATE * np.ones((sim_config['n'])), np.ones((sim_config['n'], sim_config['T'],NUM_MONTE_CARLO_ATE)), use_sigmoid=True)

    all_0_mean = np.mean(sim_results_0["rewards"])
    all_1_mean = np.mean(sim_results_1["rewards"])

    print("="*60+ "\nTRUE ATE (approximated using Monte Carlo)\n" + "="*60)
    print(f"Mean reward under all-1 vs. all-0: {all_1_mean:.4f} vs. {all_0_mean:.4f}  ")
    true_ATE = all_1_mean - all_0_mean
    print(f"True ATE: {true_ATE:.4f}")
    
    # Step 3: Find propensity scores (using Monte Carlo)
    print("\nStep 3: Computing propensity scores...\nConfig:")
    print_nicely.print_dict(sim_config)
    arms_tensor = stats_helpers.generate_cluster_treatments(cluster_matrix,time_cluster_matrix,num_W=NUM_PROP_SCORE_SIMS)
    
    # Compute propensity scores
    emp_prop_score_results = stats_helpers.empirical_propensity_scores(arms_tensor,adj_matrix,time_adj_matrix,sim_config['delta'])
    propensity_1_array, propensity_0_array = emp_prop_score_results['propensity_1'], emp_prop_score_results['propensity_0']

    # Display propensity score
    prop_1_mean = emp_prop_score_results['propensity_1'].mean()
    prop_0_mean = emp_prop_score_results['propensity_0'].mean()
    print(f"Mean emp propensity scores: {prop_1_mean:.4f}, {prop_0_mean:.4f}")
    n, T_val = sim_config['n'], sim_config['T']
    print(f"Expected #(i,t) with X_it=1 is {n*T_val*prop_1_mean:.2f}, {n*T_val*prop_0_mean:.2f}")
    print(f"%nz in emp prop_score_0, prop_score_1: {len(np.nonzero(propensity_0_array)[0])/(n*T_val)*100:.2f}%, {len(np.nonzero(propensity_1_array)[0])/(n*T_val)*100:.2f}%")
    utils.print_time()
    
    # Step 4: HT estimator
    print("\nStep 4: Running estimator simulation...")
    num_iter_est = NUM_ITER_EST

    detailed_results = [] # Store detailed results for CSV output
    
    # Main loop
    start_time = time.time()

    arms_array = stats_helpers.generate_cluster_treatments(cluster_matrix,time_cluster_matrix,num_iter_est)
    exposure_results = stats_helpers.exposure_mapping(arms_array, adj_matrix, time_adj_matrix, sim_config['delta'])
    
    sim_results = MC.simulate_MC(INITIAL_STATE * np.ones((sim_config['n'])), arms_array, use_sigmoid=True)
    rewards = sim_results["rewards"]

    print("\nDone with simulation, computing HT and Hajek estimates")

    ht_results = stats_helpers.horvitz_thompson(rewards,exposure_results['exposure_1'],exposure_results['exposure_0'],propensity_1_array,propensity_0_array)
    ate_estimate_ht = ht_results['ate_estimate_ht']

    hajek_results = stats_helpers.hajek(rewards,exposure_results['exposure_1'],exposure_results['exposure_0'],propensity_1_array,propensity_0_array)
    ate_estimate_hajek = hajek_results['ate_estimate_hajek']

    # Calculate total simulation runtime
    total_runtime_seconds = time.time() - simulation_start_time
    
    print("\n" + "="*60 + "\nHORVITZ-THOMPSON ESTIMATES\n" + "="*60)
    mean_HT_est, var_HT_est = ate_estimate_ht.mean(), ate_estimate_ht.var()
    print(f"Mean HT estimate: {mean_HT_est:.4f}")
    print(f"Bias: {mean_HT_est - true_ATE:.4f}")
    print(f"Variance of HT estimate: {var_HT_est:.4f}")
    print(f"Standard deviation: {np.sqrt(var_HT_est):.4f}")

    print("\n" + "="*60 + "\nHAJEK ESTIMATES\n" + "="*60)
    mean_Hajek_est, var_Hajek_est = ate_estimate_hajek.mean(), ate_estimate_hajek.var()
    print(f"Mean HT estimate: {mean_Hajek_est:.4f}")
    print(f"Bias: {mean_Hajek_est - true_ATE:.4f}")
    print(f"Variance of HT estimate: {var_Hajek_est:.4f}")
    print(f"Standard deviation: {np.sqrt(var_Hajek_est):.4f}")

    print(f"Total simulation runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")
    utils.print_time()

    print("\nPreparing Log Files and Saving Data, Printing results from a subset of Iterations")

    # This loop is purely to print logs to the file and terminal in order to be somewhat backwards compatible in debugging
    # But the calculation is already done so we could also skip the below loop and just save the above tensorized results dirrectly
    for iter_idx in range(num_iter_est):
        est_result={'ate_estimate_ht': ate_estimate_ht[iter_idx],
                    'ate_estimate_hajek': ate_estimate_hajek[iter_idx],
                'exposure_1': exposure_results['exposure_1'][:,:,iter_idx],
                'exposure_0': exposure_results['exposure_0'][:,:,iter_idx],
                'rewards': sim_results["rewards"][:,:,iter_idx]
        }

        # Store detailed results for this iteration with readable column names
        iteration_result = {
            'Iteration_Number': iter_idx + 1,  # 1-indexed for readability
            'HT_Estimate': est_result['ate_estimate_ht'],
            'Hajek_Estimate': est_result['ate_estimate_hajek'],
            'True_ATE': true_ATE,
            'Bias_HT': est_result['ate_estimate_ht'] - true_ATE,
            'Bias_Hajek': est_result['ate_estimate_hajek'] - true_ATE,
            'Mean_Reward_Treatment_Group': est_result['rewards'][np.nonzero(est_result['exposure_1'])].mean() if len(np.nonzero(est_result['exposure_1'])[0]) > 0 else 0,
            'Mean_Reward_Control_Group': est_result['rewards'][np.nonzero(est_result['exposure_0'])].mean() if len(np.nonzero(est_result['exposure_0'])[0]) > 0 else 0,
            'Number_Treatment_Units': len(np.nonzero(est_result['exposure_1'])[0]),
            'Number_Control_Units': len(np.nonzero(est_result['exposure_0'])[0]),
            'Reward_Difference': (est_result['rewards'][np.nonzero(est_result['exposure_1'])].mean() if len(np.nonzero(est_result['exposure_1'])[0]) > 0 else 0) - (est_result['rewards'][np.nonzero(est_result['exposure_0'])].mean() if len(np.nonzero(est_result['exposure_0'])[0]) > 0 else 0),
            'Iteration_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        detailed_results.append(iteration_result)
        
        # Progress monitoring: Display elapsed time, and ETA
        print_nicely.print_progress_ht(
            iter=iter_idx,
            num_iter_simulation=num_iter_est, 
            start_time=start_time,
            est_result=est_result,
            rewards_array=sim_results["rewards"][:,:,iter_idx],
            ht_result=ate_estimate_ht,
            true_ATE=true_ATE,
            print_freq=int(num_iter_est/5)
        )    

    # Save results with human-readable column names
    summary_results = {
        **config_data,
        'TRUE_ATE': true_ATE,
        'All_Treatment_Mean_Reward': all_1_mean,
        'All_Control_Mean_Reward': all_0_mean,
        'Propensity_Score_Treatment_Mean': prop_1_mean,
        'Propensity_Score_Control_Mean': prop_0_mean,
        'HT_Estimate_Mean': mean_HT_est,
        'HT_Bias': mean_HT_est - true_ATE,
        'HT_Variance': var_HT_est,
        'HT_Standard_Deviation': np.sqrt(var_HT_est),
        'Hajek_Estimate_Mean': mean_Hajek_est,
        'Hajek_Bias': mean_Hajek_est - true_ATE,
        'Hajek_Variance': var_Hajek_est,
        'Hajek_Standard_Deviation': np.sqrt(var_Hajek_est),
        'Total_Network_Nodes': N,
        'Expected_Treatment_Exposures': n*T_val*prop_1_mean,
        'Expected_Control_Exposures': n*T_val*prop_0_mean,
        'Total_Runtime_Seconds': total_runtime_seconds,
        'Total_Runtime_Minutes': total_runtime_seconds/60,
        'Simulation_Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Simulation_Date': datetime.now().strftime('%Y-%m-%d'),
        'Completed_At_NY': datetime.now().strftime('%H:%M:%S EST')
    }
    # Save results with useful variables for debugging
    additional_vars = {
        # 'baseline_array': baseline_array,
        # 'slope_array': slope_array,
        # 'laziness_array': laziness_array,
        # 'alpha_array': alpha_array,
        # 'beta_array': beta_array,
        # 'gamma_array': gamma_array,
        # 'ate_monte_carlo_results': ate_monte_carlo,
        'emp_prop_score_results': emp_prop_score_results
    }
    
    # Let utils create the timestamped folder and save all other results
    saved_run_folder = utils.save_results_to_csv(
        summary_results=summary_results, 
        detailed_results=detailed_results, 
        output_dir=OUTPUT_DIR, 
        filename=OUTPUT_FILENAME,
        ht_result=ate_estimate_ht,
        hajek_result=ate_estimate_hajek,
        propensity_1_array=propensity_1_array,
        propensity_0_array=propensity_0_array,
        adj_matrix=adj_matrix,
        additional_vars=additional_vars
    )
    
    # Restore stdout and close temporary log file
    sys.stdout = original_stdout
    logger.close()
    
    # Move log file to the correct results folder
    final_log_path = os.path.join(saved_run_folder, "_simulation_log.txt")
    import shutil
    shutil.move(temp_log_path, final_log_path)
    
    print(f"Simulation completed. Log saved to: {final_log_path}")
    
    return summary_results, detailed_results, saved_run_folder


if __name__ == "__main__":
    """Run the simulation when script is executed directly"""
    print("Starting Clustered Switchback Experiment Simulation...")
    print(f"Timestamp: {datetime.now()}")
    
    try:
        summary_results, detailed_results, run_folder = main()
        print("\n" + "="*80)
        print("SIMULATION SUMMARY")
        print("="*80)
        print(f"True ATE: {summary_results['TRUE_ATE']:.4f}")
        print(f"Mean HT Estimate: {summary_results['HT_Estimate_Mean']:.4f}")
        print(f"HT Bias: {summary_results['HT_Bias']:.4f}")
        print(f"HT Standard Deviation: {summary_results['HT_Standard_Deviation']:.4f}")
        print(f"Mean Hajek Estimate: {summary_results['Hajek_Estimate_Mean']:.4f}")
        print(f"Hajek Bias: {summary_results['Hajek_Bias']:.4f}")
        print(f"Hajek Standard Deviation: {summary_results['Hajek_Standard_Deviation']:.4f}")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise
