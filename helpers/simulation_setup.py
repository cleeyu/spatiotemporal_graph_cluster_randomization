import time, importlib, sys, os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import pickle

from helpers import utils, graph_helpers, mdp_helpers, stats_helpers, print_nicely

def setup_Hotel_Dataset(filename, quantization):

    df = pd.read_csv(filename)
    coord = df[['XCOORD', 'YCOORD']].values
    capacities = np.array(df[['NUMROOMS']].values).flatten()

    # scale the x,y coordinates to be within the unit square.
    scale = max(np.max(coord[:, 0]) - np.min(coord[:, 0]), np.max(coord[:, 1]) - np.min(coord[:, 1]))
    print(scale)
    coord[:,0] = (coord[:,0] - np.min(coord[:,0]))/scale
    coord[:,1] = (coord[:,1] - np.min(coord[:,1]))/scale

    # For entries that did not have capacity, we randomly sample without replacement from the distribution of reported capacitites
    permuted_capacities = np.random.permutation(capacities[~np.isnan(capacities)])
    print(permuted_capacities[0:np.sum(np.isnan(capacities))])
    capacities[np.isnan(capacities)] = permuted_capacities[0:np.sum(np.isnan(capacities))]

    capacities = np.ceil(capacities / quantization)

    return (coord, capacities)

def setup_MC_model(adj_matrix, T, max_inventory, initial_state, save_f = None):
    n = adj_matrix.shape[0]

    # Generate the Markov Chain 
    C_baseline = np.outer(np.random.random(n), np.ones(T))
    C_slope = np.ones((n,T)) + 0.2 * np.random.random((n, T))
    C_lazy = 0.1 * np.ones((n,T))
    C_alpha = np.random.normal(loc = -5, scale = 2, size = ((n,T)))
    C_beta = np.random.normal(loc = 10, scale = 4, size = ((n,T)))
    C_gamma = 0.5*np.random.random((n,T)) * C_beta
    C_depart = np.random.random((n,T))
    if save_f != None:
        np.savez(save_f,C_baseline,C_slope,C_lazy,C_alpha,C_beta,C_gamma,C_depart)

    MC_model = mdp_helpers.InventoryMarkovChain(
        max_inventory=max_inventory,
        adj_matrix=adj_matrix,
        num_rounds=T,
        C_baseline = C_baseline,
        C_slope= C_slope,
        C_lazy = C_lazy,
        C_alpha = C_alpha,
        C_beta = C_beta,
        C_gamma = C_gamma,
        C_depart = C_depart,
        initial_state = initial_state)
    
    return MC_model

def load_MC_model(adj_matrix, T, max_inventory, initial_state,MC_param_filename):
    MC_parameters = np.load(MC_param_filename)

    MC_model = mdp_helpers.InventoryMarkovChain(
        max_inventory=max_inventory,
        adj_matrix=adj_matrix,
        num_rounds=T,
        C_baseline = MC_parameters['arr_0'][:,:T],
        C_slope= MC_parameters['arr_1'][:,:T],
        C_lazy = MC_parameters['arr_2'][:,:T],
        C_alpha = MC_parameters['arr_3'][:,:T],
        C_beta = MC_parameters['arr_4'][:,:T],
        C_gamma = MC_parameters['arr_5'][:,:T],
        C_depart = MC_parameters['arr_6'][:,:T],
        initial_state = initial_state)
    
    return MC_model

def simulate_experiment(ncpd,tbl,coords_array,num_rounds,MC_model,num_iter_est,save_data_folder = None):
    # Setup randomized design
    cluster_matrix = graph_helpers.spatial_clustering_map(coords_array, ncpd)
    time_cluster_matrix = stats_helpers.generate_time_blocks(num_rounds, tbl)
    
    arms_array = stats_helpers.generate_cluster_treatments(cluster_matrix,time_cluster_matrix,num_iter_est)   
    sim_results = MC_model.simulate_MC(arms_array, use_sigmoid=True)
    rewards = sim_results["rewards"]
    
    if save_data_folder != None:
        np.save(save_data_folder+f'/arms_array_{ncpd}_{tbl}.npy', arms_array)
        np.save(save_data_folder+f'/rewards_{ncpd}_{tbl}.npy', rewards)

    return (ncpd,tbl,arms_array,rewards)

def run_simulation(sim_configs, save_data_f, save_estimates_f, load_MC_param_file = None):

    # unpack variables from sim_configs
    num_rounds = sim_configs['maximum_T']
    num_cells_per_dim = sim_configs['num_cells_per_dim']
    time_block_length = sim_configs['time_block_length']
    recency = sim_configs['recency']
    delta = sim_configs['delta']
    burn_in = sim_configs['burn_in']
    num_sims_apx_GATE = sim_configs['num_monte_carlo_gate']
    num_iter_est = sim_configs['num_iter_est']
    coords_array = sim_configs['coords_array']
    kappa = sim_configs['kappa']
    max_inventory = sim_configs['max_inventory']
    initial_state = sim_configs['initial_state']

    n = coords_array.shape[0]

    start_time = time.time()
    print("\nStep 1: Setup Interference Graph and Markov Chain model\n")

    # Create interference graph
    adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, kappa)

    # Setup MC model
    if load_MC_param_file == None:
        MC_model = setup_MC_model(adj_matrix, num_rounds, max_inventory, initial_state)
    else:
        MC_model = load_MC_model(adj_matrix, num_rounds, max_inventory, initial_state, load_MC_param_file)

    # Find true GATE using Monte Carlo
    print("\nStep 2: Computing true GATE using Monte Carlo...")
    results_GATE = MC_model.estimate_GATE(num_sims_apx_GATE)
    true_GATE = results_GATE[0]
    all_1_mean = results_GATE[1]
    all_0_mean = results_GATE[2]
    std_dev_GATE = results_GATE[3]
    diff_means = results_GATE[4]

    print(f"GATE Estimate: {true_GATE:.4f}")
    print(f"Std Dev: {std_dev_GATE:.4f}")

    total_runtime_seconds = time.time() - start_time
    print(f"Elapsed time: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")

    # Sample design and run simulation
    print(f"\nStep 3: Generating treatment vectors and running estimator simulation")

    pairs = [(ncpd,tbl) for ncpd in num_cells_per_dim for tbl in time_block_length]
    joblib_results = Parallel(n_jobs=-1)(
        delayed(simulate_experiment)(ncpd,tbl,coords_array,num_rounds,MC_model,num_iter_est) for ncpd,tbl in pairs
    )
    with open(save_data_f, 'wb') as f:
        pickle.dump(joblib_results, f) #    

    total_runtime_seconds = time.time() - start_time
    print(f"Elapsed time: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")

    # Compute estimates
    print("\nStep 4:Compute estimates")
    parameter_data_combo = [(r,d,data) for r in recency for d in delta for data in joblib_results]
    HT_Hajek_results = Parallel(n_jobs=-1)(
        delayed(compute_HT_Hajek_estimates)(r,d,data,adj_matrix) for r,d,data in parameter_data_combo
    )
    all_results = pd.concat(HT_Hajek_results, join='outer', ignore_index=True)

    DM_results = Parallel(n_jobs=-1)(
        delayed(compute_DM_estimates)(burn_in,data) for data in joblib_results
    )
    all_results = pd.concat([all_results,pd.concat(DM_results, ignore_index=True)],axis=0, join='outer', ignore_index=True)

    all_results['kappa'] = kappa
    all_results['true_GATE'] = true_GATE

    total_runtime_seconds = time.time() - start_time
    print(f"Elapsed time: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")

    print_logs(all_results)
    all_results.to_csv(save_estimates_f)

    return all_results

def compute_HT_Hajek_estimates(recency,delta,data,adj_matrix, truncate = None):
    print(f"Computing HT / Hajek Estimates with recency = {recency},delta = {delta:.2f}")
    start_time = time.time()

    num_cells_per_dim = data[0]
    time_block_length = data[1]
    arms_array = data[2]
    rewards = data[3]
    if truncate != None:
        rewards = rewards[:,:truncate,:]
        arms_array = arms_array[:,:truncate,:]
    num_rounds = rewards.shape[1]

    time_adj_matrix = np.tril(np.ones((num_rounds,num_rounds)), k=0) - np.tril(np.ones((num_rounds,num_rounds)), k=-(recency + 1))
    exposure_results = stats_helpers.exposure_mapping(arms_array, adj_matrix, time_adj_matrix, delta)
    propensity_1_array = np.average(exposure_results['exposure_1'], axis=2)
    propensity_0_array = np.average(exposure_results['exposure_0'], axis=2)
    
    # print_propensity_stats(propensity_1_array, propensity_0_array)

    # # CHECK with deteriministic rewards set to be all_0_mean for all control units and all_1_mean for all treated units
    # ht_fake_results = stats_helpers.horvitz_thompson(all_0_mean * (1- arms_array) + all_1_mean * arms_array,exposure_results['exposure_1'],exposure_results['exposure_0'],propensity_1_array,propensity_0_array)
    # all_results.append({'name': 'Horvitz-Thompson(oracle rewards)', 'results': ht_fake_results})

    # HT with interference exposure mapping
    ht_results = stats_helpers.horvitz_thompson(rewards,exposure_results['exposure_1'],exposure_results['exposure_0'],propensity_1_array,propensity_0_array)
    df_HT = pd.DataFrame(ht_results)
    df_HT['name'] = f"Horvitz-Thompson(recency={recency},delta={delta:.2f})"
    df_HT['type'] = 'HT'
    df_HT['recency'] = recency
    df_HT['delta'] = delta
    df_HT['num_cells_per_dim'] = data[0]
    df_HT['time_block_length'] = data[1]
    df_HT['n'] = rewards.shape[0]
    df_HT['T'] = num_rounds
    
    # Hajek with interference exposure mapping
    hajek_results = stats_helpers.hajek(rewards,exposure_results['exposure_1'],exposure_results['exposure_0'],propensity_1_array,propensity_0_array)
    df_Hajek = pd.DataFrame(hajek_results)
    df_Hajek['name'] = f"Hajek(recency={recency},delta={delta:.2f})"
    df_Hajek['type'] = 'Hajek'
    df_Hajek['recency'] = recency
    df_Hajek['delta'] = delta
    df_Hajek['num_cells_per_dim'] = num_cells_per_dim
    df_Hajek['time_block_length'] = time_block_length
    df_Hajek['n'] = rewards.shape[0]
    df_Hajek['T'] = num_rounds

    # Calculate total simulation runtime
    total_runtime_seconds = time.time() - start_time
    print(f"Total simulation runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")
    utils.print_time()
    
    return pd.concat([df_HT,df_Hajek],axis=0, join='outer', ignore_index=True)

def compute_DM_estimates(burn_in, data,truncate = None):
    start_time = time.time()
    num_cells_per_dim = data[0]
    time_block_length = data[1]
    arms_array = data[2]
    rewards = data[3]
    if truncate != None:
        rewards = rewards[:,:truncate,:]
        arms_array = arms_array[:,:truncate,:]

    all_results = pd.DataFrame()

    # Diff in Means with burn in
    for b in burn_in:
        print(f"Computing Diff-in-Means Estimates with burn-in = {b}")
        DM_results = stats_helpers.diff_means(rewards,arms_array,time_block_length, b)
        df_DM = pd.DataFrame(DM_results)
        df_DM['name'] = f"Diff-Means(burn-in={b})"
        df_DM['type'] = 'DM'
        df_DM['burn-in'] = b
        all_results = pd.concat([all_results,df_DM],axis=0, join='outer', ignore_index=True)

    all_results['n'] = rewards.shape[0]
    all_results['T'] = rewards.shape[1]
    all_results['num_cells_per_dim'] = num_cells_per_dim
    all_results['time_block_length'] = time_block_length

    # Calculate total simulation runtime
    total_runtime_seconds = time.time() - start_time
    print(f"Total simulation runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")
    utils.print_time()
    
    return all_results

def print_propensity_stats(propensity_1_array, propensity_0_array):
    print(f"Mean emp propensity scores: {propensity_0_array.mean():.4f}, {propensity_1_array.mean():.4f}")
    print(f"Expected #(i,t) with X_it=1 is {propensity_0_array.sum():.2f}, {propensity_1_array.sum():.2f}")
    print(f"%nz in emp prop_score_0, prop_score_1: {len(np.nonzero(propensity_0_array)[0])/(propensity_0_array.size)*100:.2f}%, {len(np.nonzero(propensity_1_array)[0])/(propensity_1_array.size)*100:.2f}%")

def print_logs(all_results):
    aggregate_stats = all_results.groupby(['name','num_cells_per_dim', 'time_block_length']).agg({'gate_estimate': ['mean', 'std'], 'true_GATE': ['mean']})
    aggregate_stats[('gate_estimate','bias')] = aggregate_stats[('gate_estimate','mean')] - aggregate_stats[('true_GATE','mean')]

    print(aggregate_stats)
