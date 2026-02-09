import time, pytz, os, pickle, json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime

# === save ===
def _create_human_readable_config(summary_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create human-readable configuration structure."""
    return {
        "Simulation Settings": {
            "Total Nodes": summary_results['Total_Network_Nodes'],
            "Time Periods": summary_results['Time_Periods_T'],
            "NumClusters Per Dimension": summary_results['Num_Clusters_Per_Dim'],
            "Time Block Length": summary_results['Time_Block_Length']
        },
        "Interference & Exposure": {
            "Kappa (Interference)": summary_results['Interference_Kappa'],
            "Delta Parameter": summary_results['Delta_Parameter'],
            "Recency Parameter": summary_results['Recency_Parameter']
        },
        "MDP Parameters": {
            "States": summary_results['MDP_States'],
            "Initial State": summary_results['Initial_MDP_State'],
            "Laziness (C_lazy)": summary_results['MDP_Laziness_C_lazy'],
            "Transition Alpha": summary_results['Transition_Alpha_C_alpha'],
            "Transition Beta": summary_results['Transition_Beta_C_beta'],
            "Transition Gamma": summary_results['Transition_Gamma_C_gamma'],
            "Reward Baseline": summary_results['Reward_Baseline_C_baseline'],
            "Reward Slope": summary_results['Reward_Slope_C_slope']
        },
        "Simulation Iterations": {
            "Monte Carlo ATE": summary_results['Monte_Carlo_ATE_Iterations'],
            "Propensity Score Sims": summary_results['Propensity_Score_Simulations'],
            "HT Estimator Iterations": summary_results['HT_Estimator_Iterations']
        },
        "Results Summary": {
            "True ATE": summary_results['TRUE_ATE'],
            "HT Estimate Mean": summary_results['HT_Estimate_Mean'],
            "Bias": summary_results['HT_Bias'],
            "Standard Deviation": summary_results['HT_Standard_Deviation']
        },
        "Simulation Info": {
            "Date": summary_results['Simulation_Date'],
            "Time": summary_results['Completed_At_NY'],
            "Full Timestamp": summary_results['Simulation_Timestamp']
        },
        "Runtime": {
            "Total Seconds": summary_results.get('Total_Runtime_Seconds', 'Not recorded'),
            "Total Minutes": summary_results.get('Total_Runtime_Minutes', 'Not recorded'),
            "Formatted": f"{summary_results.get('Total_Runtime_Seconds', 0):.2f} seconds ({summary_results.get('Total_Runtime_Minutes', 0):.2f} minutes)" if 'Total_Runtime_Seconds' in summary_results else "Not recorded"
        }
    }

def _save_config_files(
        run_folder: str, 
        base_name: str, 
        summary_results: Dict[str, Any]
        ) -> None:
    """Save configuration files (JSON and CSV)."""
    # Human-readable JSON config
    human_config = _create_human_readable_config(summary_results)
    config_json_path = os.path.join(run_folder, "config_readable.json")
    with open(config_json_path, 'w') as f:
        json.dump(human_config, f, indent=4)
    print(f"Human-readable config saved to: {config_json_path}")
    
    # Complete config CSV (for compatibility)
    config_path = os.path.join(run_folder, f"{base_name}_config.csv")
    config_df = pd.DataFrame([summary_results])
    config_df.to_csv(config_path, index=False)
    print(f"Full configuration saved to: {config_path}")

def _save_result_csvs(
        run_folder: str, 
        base_name: str, 
        summary_results: Dict[str, Any], 
        detailed_results: Optional[List[Dict[str, Any]]]
        ) -> None:
    """Save main and detailed results CSV files."""
    # Main results CSV (concise summary)
    main_results = {
        'Time_Periods': summary_results['Time_Periods_T'],
        'HT_Iterations': summary_results['HT_Estimator_Iterations'], 
        'Interference_Kappa': summary_results['Interference_Kappa'],
        'Recency': summary_results['Recency_Parameter'],
        'TRUE_ATE': summary_results['TRUE_ATE'],
        'HT_Estimate_Mean': summary_results['HT_Estimate_Mean'],
        'HT_Bias': summary_results['HT_Bias'],
        'HT_Standard_Deviation': summary_results['HT_Standard_Deviation'],
        'Total_Nodes': summary_results['Total_Network_Nodes'],
        'Runtime_Seconds': summary_results.get('Total_Runtime_Seconds', 'Not recorded'),
        'Runtime_Minutes': summary_results.get('Total_Runtime_Minutes', 'Not recorded'),
        'Simulation_Date': summary_results['Simulation_Date'],
        'Completed_At_NY': summary_results.get('Completed_At_NY', 'Not recorded')
    }
    main_df = pd.DataFrame([main_results])
    main_path = os.path.join(run_folder, f"{base_name}_main.csv")
    main_df.to_csv(main_path, index=False)
    print(f"Main results saved to: {main_path}")
    
    # Detailed iteration results
    if detailed_results and len(detailed_results) > 1:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(run_folder, f"{base_name}_iterations.csv")
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Iteration details saved to: {detailed_path}")


def _save_variables(
        run_folder: str, 
        ht_result: Optional[np.ndarray], 
        hajek_result: Optional[np.ndarray], 
        propensity_1_array: Optional[np.ndarray], 
        propensity_0_array: Optional[np.ndarray],
        adj_matrix: Optional[np.ndarray], 
        additional_vars: Optional[Dict[str, Any]]
        ) -> List[str]:
    """Save useful variables for debugging and return list of saved items."""
    variables_saved = []
    
    if ht_result is not None:
        ht_path = os.path.join(run_folder, "ht_result_array.npy")
        np.save(ht_path, ht_result)
        variables_saved.append(f"ht_result ({len(ht_result)} values)")
        print(f"HT result array saved to: {ht_path}")
    
    if hajek_result is not None:
        hajek_path = os.path.join(run_folder, "hajek_result_array.npy")
        np.save(hajek_path, hajek_result)
        variables_saved.append(f"hajek_result ({len(hajek_result)} values)")
        print(f"HAJEK result array saved to: {hajek_path}")
    
    if propensity_1_array is not None:
        prop1_path = os.path.join(run_folder, "propensity_1_array.npy")
        np.save(prop1_path, propensity_1_array)
        variables_saved.append(f"propensity_1_array {propensity_1_array.shape}")
        print(f"Propensity 1 array saved to: {prop1_path}")
    
    if propensity_0_array is not None:
        prop0_path = os.path.join(run_folder, "propensity_0_array.npy")
        np.save(prop0_path, propensity_0_array)
        variables_saved.append(f"propensity_0_array {propensity_0_array.shape}")
        print(f"Propensity 0 array saved to: {prop0_path}")
    
    if adj_matrix is not None:
        adj_path = os.path.join(run_folder, "adjacency_matrix.npy")
        np.save(adj_path, adj_matrix)
        variables_saved.append("adj_matrix (network graph)")
        print(f"Adjacency matrix saved to: {adj_path}")
    
    if additional_vars:
        for var_name, var_data in additional_vars.items():
            var_path = os.path.join(run_folder, f"{var_name}.pkl")
            with open(var_path, 'wb') as f:
                pickle.dump(var_data, f)
            variables_saved.append(var_name)
            print(f"Variable '{var_name}' saved to: {var_path}")
    
    return variables_saved


def _save_readme(
        run_folder: str, 
        base_name: str, 
        timestamp: str, 
        summary_results: Dict[str, Any], 
        variables_saved: List[str]
        ) -> None:
    """Save README file with folder summary."""
    summary_info = {
        "Run Information": {
            "Folder": run_folder,
            "Timestamp": timestamp,
            "Network Size": f"{summary_results['Total_Network_Nodes']} nodes",
            "HT Iterations": summary_results['HT_Estimator_Iterations']
        },
        "Files Saved": {
            "config_readable.json": "Human-readable configuration",
            f"{base_name}_main.csv": "Summary results",
            f"{base_name}_iterations.csv": "Detailed iteration results",
            f"{base_name}_config.csv": "Complete configuration (CSV)",
        },
        "Variables Saved": variables_saved,
        "Quick Results": {
            "True ATE": summary_results['TRUE_ATE'],
            "HT Estimate": summary_results['HT_Estimate_Mean'],
            "Bias": summary_results['HT_Bias']
        }
    }
    
    summary_path = os.path.join(run_folder, "README.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_info, f, indent=4)
    print(f"Folder summary saved to: {summary_path}")


def save_results_to_csv(
        summary_results: Dict[str, Any], 
        detailed_results: Optional[List[Dict[str, Any]]],                
        output_dir: str, 
        filename: str, 
        ht_result: Optional[np.ndarray] = None, 
        hajek_result: Optional[np.ndarray] = None, 
        propensity_1_array: Optional[np.ndarray] = None, 
        propensity_0_array: Optional[np.ndarray] = None, 
        adj_matrix: Optional[np.ndarray] = None, 
        additional_vars: Optional[Dict[str, Any]] = None
        ) -> str:
    """ Save simulation results to files in timestamped subfolder. """
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    
    base_name = filename.replace('.csv', '')
    
    # Save all file types
    _save_config_files(run_folder, base_name, summary_results)
    _save_result_csvs(run_folder, base_name, summary_results, detailed_results)
    
    variables_saved = _save_variables(run_folder, ht_result, hajek_result, propensity_1_array, propensity_0_array, adj_matrix, additional_vars)
    
    _save_readme(run_folder, base_name, timestamp, summary_results, variables_saved)
    
    print(f"\nAll results saved in folder: {run_folder}.\nVariables saved: {len(variables_saved)} items")
    return run_folder


# === Load ===

def load_simulation_results(subfolder_name: str) -> Optional[Dict[str, Any]]:
    """
    Load all simulation results from a specified subfolder in the results directory.
    Args: subfolder_name (str): Name of the subfolder in results/ (e.g., 'run_20260127_202052')
    Returns: dict: Dictionary containing all loaded data with filenames as keys
    """
    results_path = os.path.join('results', subfolder_name)
    if not os.path.exists(results_path):
        print(f"Error: Subfolder '{subfolder_name}' not found in results/")
        return None
    loaded_data = {}
    
    # Get all files in the subfolder
    for filename in os.listdir(results_path):
        filepath = os.path.join(results_path, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
        try:
            # Load based on file extension
            if filename.endswith('.pkl'):
                with open(filepath, 'rb') as f:
                    loaded_data[filename] = pickle.load(f)
                print(f"✓ Loaded pickle file: {filename}")
            elif filename.endswith('.npy'):
                loaded_data[filename] = np.load(filepath)
                print(f"✓ Loaded numpy array: {filename}")
            elif filename.endswith('.csv'):
                loaded_data[filename] = pd.read_csv(filepath)
                print(f"✓ Loaded CSV file: {filename}")
            elif filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    loaded_data[filename] = json.load(f)
                print(f"✓ Loaded JSON file: {filename}") 
            else:
                print(f"⚠ Skipped unknown file type: {filename}")
                
        except Exception as e:
            print(f"✗ Error loading {filename}: {str(e)}")
    
    print(f"\n📊 Successfully loaded {len(loaded_data)} files from {subfolder_name}")
    return loaded_data

# === Other ===

def print_time() -> None:
    """ Print the current completion time in New York timezone. """
    ny_tz = pytz.timezone('America/New_York')
    ts = datetime.now(ny_tz)
    print(f"Printed at: {ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    """ Sigmoid function to map values to [0,1] """
    return 1 / (1 + np.exp(-x))

