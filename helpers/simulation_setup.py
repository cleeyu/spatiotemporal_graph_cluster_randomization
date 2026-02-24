import time, importlib, sys, os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

from helpers import utils, graph_helpers, mdp_helpers, stats_helpers, print_nicely

def setup_MC_model(adj_matrix, T, max_inventory, initial_state):
    n = adj_matrix.shape[0]

    # Generate the Markov Chain 
    C_baseline = np.outer(np.random.random(n), np.ones(T))
    C_slope = np.ones((n,T)) + 0.2 * np.random.random((n, T))
    C_lazy = 0.1 * np.ones((n,T))
    C_alpha = np.random.normal(loc = -5, scale = 2, size = ((n,T)))
    C_beta = np.random.normal(loc = 10, scale = 4, size = ((n,T)))
    C_gamma = 0.5*np.random.random((n,T)) * C_beta
    C_depart = np.random.random((n,T))

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
        initial_state = initial_state * np.ones(n))
    
    return MC_model

def run_simulation(sim_config,num_sims_apx_ATE,num_iter_est):
    print_nicely.print_dict(sim_config)

    n = sim_config['n'] 
    kappa = sim_config['kappa']
    num_rounds = sim_config['T']

    print("\nStep 1: Setup Interference Graph and Markov Chain model")

    # Create interference graph
    coords_array = graph_helpers.generate_random_points(n)
    adj_matrix = graph_helpers.build_adjacency_matrix_from_coords(coords_array, kappa)

    # Setup MC model
    max_inventory=(sim_config['num_states']-1)*np.ones(n)
    initial_state = sim_config['initial_state']*np.ones(n)
    MC_model = setup_MC_model(adj_matrix, num_rounds, max_inventory, initial_state)

    # Find true ATE using Monte Carlo
    print("\nStep 2: Computing true ATE using Monte Carlo...")
    start_time = time.time()
    (true_ATE, all_1_mean, all_0_mean) = MC_model.estimate_GATE(num_sims_apx_ATE)
    
    # Step 3: Sample design and run smulation
    print("\nStep 3: Running estimator simulation...")
    # Setup randomized design
    cluster_matrix = graph_helpers.spatial_clustering_map(coords_array, sim_config['num_cells_per_dim'])
    time_cluster_matrix = stats_helpers.generate_time_blocks(num_rounds, sim_config['time_block_length'])
    
    arms_array = stats_helpers.generate_cluster_treatments(cluster_matrix,time_cluster_matrix,num_iter_est)   
    sim_results = MC_model.simulate_MC(arms_array, use_sigmoid=True)
    rewards = sim_results["rewards"]

    # Step 4: Compute estimates
    print("\nStep 4:Compute estimates")

    all_results = pd.DataFrame()

    # for a combo of recency and delta
    recency = [0,2,4,6]
    delta = [0, 0.1, 0.2, 0.3]
    for r in recency:
        for d in delta:
            df_results = compute_HT_Hajek_estimates(rewards, arms_array, adj_matrix, r, d)
            all_results = pd.concat([all_results,df_results],axis=0, join='outer', ignore_index=True)

    burn_in = [0,2,4,6]
    df_results = compute_DM_estimates(rewards, arms_array, sim_config['time_block_length'], burn_in)
    all_results = pd.concat([all_results,df_results],axis=0, join='outer', ignore_index=True)

    print_logs(all_results, true_ATE)

    return all_results

def compute_HT_Hajek_estimates(rewards, arms_array, adj_matrix, recency, delta):
    print(f"Computing HT / Hajek Estimates with recency = {recency},delta = {delta:.2f}")
    start_time = time.time()
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
 
    # Hajek with interference exposure mapping
    hajek_results = stats_helpers.hajek(rewards,exposure_results['exposure_1'],exposure_results['exposure_0'],propensity_1_array,propensity_0_array)
    df_Hajek = pd.DataFrame(hajek_results)
    df_Hajek['name'] = f"Hajek(recency={recency},delta={delta:.2f})"
    df_Hajek['type'] = 'Hajek'
    df_Hajek['recency'] = recency
    df_Hajek['delta'] = delta

    # Calculate total simulation runtime
    total_runtime_seconds = time.time() - start_time
    print(f"Total simulation runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")
    utils.print_time()
    
    return pd.concat([df_HT,df_Hajek],axis=0, join='outer', ignore_index=True)

def compute_DM_estimates(rewards, arms_array, time_block_length, burn_in):
    start_time = time.time()

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

    # Calculate total simulation runtime
    total_runtime_seconds = time.time() - start_time
    print(f"Total simulation runtime: {total_runtime_seconds:.2f} seconds ({total_runtime_seconds/60:.2f} minutes)")
    utils.print_time()
    
    return all_results

def print_propensity_stats(propensity_1_array, propensity_0_array):
    print(f"Mean emp propensity scores: {propensity_0_array.mean():.4f}, {propensity_1_array.mean():.4f}")
    print(f"Expected #(i,t) with X_it=1 is {propensity_0_array.sum():.2f}, {propensity_1_array.sum():.2f}")
    print(f"%nz in emp prop_score_0, prop_score_1: {len(np.nonzero(propensity_0_array)[0])/(propensity_0_array.size)*100:.2f}%, {len(np.nonzero(propensity_1_array)[0])/(propensity_1_array.size)*100:.2f}%")

def print_logs(all_results, true_ATE):
    # all_results.to_csv('all_results.csv')
    aggregate_stats = all_results.groupby('name').agg({'gate_estimate': ['mean', 'std']})
    aggregate_stats[('gate_estimate','bias')] = aggregate_stats[('gate_estimate','mean')] - true_ATE

    print(aggregate_stats)
