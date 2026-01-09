import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

PLOT_DIR = "histogram_plots"
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Saving plots to {os.path.abspath(PLOT_DIR)}/")

def plot_distri(model_dir):
    
    # --- 2. Find Model Files ---
    model_files = glob.glob(os.path.join(model_dir, "*_PR_evol_onehot_logistic.p"))
    if not model_files:
        print(f"Error: No model files found in directory: {model_dir}")
        print("Please ensure your .p files are in that folder.")
        return
        

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

        all_interactions = []
        
        # Iterate over each position 'j' (the one being predicted)
        for j, model in models.items():
            coef_matrix = model.coef_

            all_interactions.append(coef_matrix.flatten())

        
        # Convert to numpy array for sorting
        interactions_arr = np.abs(np.hstack(all_interactions))

        n_interactions = len(interactions_arr)

        plt.title(f'{treatment_name}, total interactions {n_interactions}', fontsize=20)
        plt.hist(interactions_arr, bins=20, color='blue')

        plt.xlabel('interaction magnitude', fontsize=20)
        plt.ylabel('counts', fontsize=20)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        plt.yscale('log')
        plt.tight_layout()

        # Save the figure
        output_filename = os.path.join(PLOT_DIR, f"{treatment_name.replace('/', '_')}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to save memory

        

    print(f"\nSuccessfully saved {len(model_files)} scatter plots to {PLOT_DIR}/")

if __name__ == "__main__":
    # Ensure you have pandas, seaborn, matplotlib, tqdm, and scipy installed
    # pip install matplotlib tqdm
    
    MODEL_DIR = "../../train_models/models/" # ASSUMES your .p files are in a 'models' subfolder
    plot_distri(MODEL_DIR)


