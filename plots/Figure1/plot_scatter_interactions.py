import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def plot_scatter_per_treatment(dist_matrix_file, model_dir, n_residues=99, n_top=5):
    """
    Loads all logistic regression models, extracts every individual parameter
    magnitude, and correlates it with the spatial distance.
    
    Generates a separate scatter plot for each treatment, highlighting
    the top N strongest interactions.
    
    Requires: numpy, matplotlib, tqdm
    """
    
    # --- 1. Load Distance Matrix ---
    if not os.path.exists(dist_matrix_file):
        print(f"Error: Distance matrix file not found: {dist_matrix_file}")
        print("Please run the 'create_distance_matrix.py' script first.")
        return
        
    print(f"Loading distance matrix from {dist_matrix_file}...")
    dist_matrix = np.load(dist_matrix_file)
    
    # --- 2. Find Model Files ---
    model_files = glob.glob(os.path.join(model_dir, "*_PR_evol_onehot_logistic.p"))
    if not model_files:
        print(f"Error: No model files found in directory: {model_dir}")
        print("Please ensure your .p files are in that folder.")
        return
        
    print(f"Found {len(model_files)} model files. Processing...")
    
    # --- 3. Create Output Directory ---
    plot_dir = "scatter_plots"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to {os.path.abspath(plot_dir)}/")

    # --- 4. Process Each Treatment ---
    for f_path in tqdm(model_files, desc="Processing Treatments"):
        try:
            fname = os.path.basename(f_path)
            treatment_name = fname.split('_PR_evol_onehot_logistic.p')[0].replace('treatment', '')
        except Exception:
            treatment_name = os.path.basename(f_path)

        # This list will hold (abs(param_value), distance)
        treatment_interactions = []
        
        try:
            with open(f_path, 'rb') as dbfile:
                info = pickle.load(dbfile)
        except Exception as e:
            print(f"Warning: Could not load or process file {f_path}. Skipping. Error: {e}")
            continue

        encoder = info['onehot_encoder']
        models = info['logistic_regression']
        
        # Get the feature boundaries
        lenns = [len(cat) for cat in encoder.categories_]
        boundaries = np.hstack([[0], np.cumsum(lenns)])
        
        # Iterate over each position 'j' (the one being predicted)
        for j, model in models.items():
            coef_matrix = model.coef_
            
            # Iterate over each position 'i' (the one used as a feature)
            for i in range(n_residues):
                if i == j:  # Skip self-interactions
                    continue
                    
                distance = dist_matrix[i, j]
                
                if np.isnan(distance): # Skip missing residues
                    continue
                
                # Find the block of features in the coef_matrix for position 'i'
                start = boundaries[i]
                end = boundaries[i+1]
                
                param_block = coef_matrix[:, start:end]
                
                # Add every single parameter magnitude and its associated distance
                for param_magnitude in np.abs(param_block.flat):
                    if param_magnitude > 1e-9: # Avoid zero values for log plot
                        treatment_interactions.append((param_magnitude, distance))

        if not treatment_interactions:
            print(f"Warning: No interactions found for treatment {treatment_name}. Skipping.")
            continue
            
        # --- 5. Prepare Data for Plotting ---
        
        # Convert to numpy array for sorting
        interactions_arr = np.array(treatment_interactions)
        
        # Sort by magnitude (column 0) in descending order
        sorted_indices = np.argsort(interactions_arr[:, 0])[::-1]
        sorted_interactions = interactions_arr[sorted_indices]
        
        # Split into top N and the rest
        top_n_points = sorted_interactions[:n_top]
        other_points = sorted_interactions[n_top:]
        
        # --- 6. Create Plot ---
        plt.figure(figsize=(12, 8))
        
        # Plot the "other" points
        plt.scatter(
            other_points[:, 0], 
            other_points[:, 1], 
            color='blue', 
            alpha=0.3, 
            label=f'Other Interactions',
            s=75 # Smaller size for the background
        )
        
        # Plot the "top N" points
        plt.scatter(
            top_n_points[:, 0], 
            top_n_points[:, 1], 
            color='red', 
            s=100, # Larger size
            edgecolor='black',
            label=f'Top {n_top} Interactions',
            zorder=3 # Plot on top
        )
        
        #plt.xscale('log') # Use log scale for magnitude
        plt.title(f"Treatment: {treatment_name}", fontsize=28)
        plt.xlabel("Absolute Interaction Magnitude", fontsize=28)
        plt.ylabel("Cα-Cα Distance (Å)", fontsize=28)
        plt.legend(fontsize=24)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Save the figure
        output_filename = os.path.join(plot_dir, f"scatter_{treatment_name.replace('/', '_')}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to save memory

    print(f"\nSuccessfully saved {len(model_files)} scatter plots to {plot_dir}/")

if __name__ == "__main__":
    # Configuration
    DIST_MATRIX_FILE = "hiv_protease_1HHP_A_dist_matrix.npy"
    MODEL_DIR = "../../train_models/models/" # ASSUMES your .p files are in a 'models' subfolder
    
    # Ensure you have pandas, seaborn, matplotlib, tqdm, and scipy installed
    # pip install matplotlib tqdm
    
    plot_scatter_per_treatment(DIST_MATRIX_FILE, MODEL_DIR)
