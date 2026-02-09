import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os


def save_propensity_arrays(propensity_1_array, propensity_0_array, 
                          filename_prefix="propensity_scores", 
                          save_format="csv", 
                          output_dir="./"):
    """
    Save propensity_1_array and propensity_0_array together in various formats.
    
    Parameters:
    -----------
    propensity_1_array : np.ndarray
        Propensity scores for treatment arm 1
    propensity_0_array : np.ndarray  
        Propensity scores for treatment arm 0
    filename_prefix : str
        Prefix for the output filename
    save_format : str
        Format to save data ('csv', 'pickle', 'npz', 'excel')
    output_dir : str
        Directory to save the file
    
    Returns:
    --------
    dict : Information about saved files
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    if save_format.lower() == "csv":
        # Method 1: Save as separate CSV files
        file_1 = os.path.join(output_dir, f"{filename_prefix}_arm1_{timestamp}.csv")
        file_0 = os.path.join(output_dir, f"{filename_prefix}_arm0_{timestamp}.csv")
        
        pd.DataFrame(propensity_1_array).to_csv(file_1, index=False)
        pd.DataFrame(propensity_0_array).to_csv(file_0, index=False)
        
        saved_files = {
            "propensity_1_file": file_1,
            "propensity_0_file": file_0,
            "format": "csv"
        }
        
        # Method 2: Also save as combined long format CSV
        combined_file = os.path.join(output_dir, f"{filename_prefix}_combined_{timestamp}.csv")
        df_combined = create_combined_dataframe(propensity_1_array, propensity_0_array)
        df_combined.to_csv(combined_file, index=False)
        saved_files["combined_file"] = combined_file
        
    elif save_format.lower() == "pickle":
        # Save as pickle (preserves exact numpy arrays)
        pickle_file = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.pkl")
        data_dict = {
            'propensity_1_array': propensity_1_array,
            'propensity_0_array': propensity_0_array,
            'metadata': {
                'shape': propensity_1_array.shape,
                'created_at': datetime.now().isoformat(),
                'arrays_equal_shape': propensity_1_array.shape == propensity_0_array.shape
            }
        }
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
        saved_files = {
            "pickle_file": pickle_file,
            "format": "pickle"
        }
        
    elif save_format.lower() == "npz":
        # Save as numpy compressed format
        npz_file = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.npz")
        np.savez_compressed(npz_file, 
                          propensity_1=propensity_1_array,
                          propensity_0=propensity_0_array)
        
        saved_files = {
