import numpy as np
import pickle
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import stats # Import for Mann-Whitney U test

def get_pi_count(treatment_name):
    """Helper function to determine the sort order of treatments."""
    if treatment_name == 'None':
        return 0
    # Check for ',' as the delimiter
    if ',' in treatment_name:
        return len(treatment_name.split(','))
    # Otherwise, it's a single PI
    return 1

def analyze_interactions(dist_matrix_file, model_dir, n_residues=99):
    """
    Loads all logistic regression models, extracts every individual parameter
    magnitude, and correlates it with the spatial distance between the
    corresponding residue pair.
    
    Generates a boxplot comparing the distance distributions for the
    Top 1% vs. Remaining 99% of interaction magnitudes.
    
    Also computes a Mann-Whitney U test for each treatment and adds the
    p-value to the plot.
    
    Requires: numpy, pandas, seaborn, matplotlib, tqdm, scipy
    """
    
    # Load the distance matrix
    if not os.path.exists(dist_matrix_file):
        print(f"Error: Distance matrix file not found: {dist_matrix_file}")
        print("Please run the 'create_distance_matrix.py' script first.")
        return
        
    print(f"Loading distance matrix from {dist_matrix_file}...")
    dist_matrix = np.load(dist_matrix_file)
    
    # Find all model files
    model_files = glob.glob(os.path.join(model_dir, "*_PR_evol_onehot_logistic.p"))
    if not model_files:
        print(f"Error: No model files found in directory: {model_dir}")
        print("Please ensure your .p files are in that folder.")
        return
        
    print(f"Found {len(model_files)} model files. Processing...")
    
    all_plot_data = []
    p_values = {} # To store p-values for each treatment

    for f_path in tqdm(model_files, desc="Processing Treatments"):
        try:
            # Extract treatment name from filename
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
            
            # model.coef_ shape is (n_classes_j, n_total_features)
            coef_matrix = model.coef_
            
            # Iterate over each position 'i' (the one used as a feature)
            for i in range(n_residues):
                if i == j:  # Skip self-interactions
                    continue
                    
                # Get the spatial distance
                distance = dist_matrix[i, j]
                
                # Skip if distance is NaN (e.g., missing residue)
                if np.isnan(distance):
                    continue
                
                # Find the block of features in the coef_matrix for position 'i'
                start = boundaries[i]
                end = boundaries[i+1]
                
                # Extract the sub-matrix of parameters
                param_block = coef_matrix[:, start:end]
                
                # Add every single parameter magnitude and its associated
                # (i, j) spatial distance to our list
                for param_magnitude in np.abs(param_block.flat):
                    treatment_interactions.append((param_magnitude, distance))

        if not treatment_interactions:
            print(f"Warning: No interactions found for treatment {treatment_name}. Skipping.")
            continue
            
        # --- Analysis for this treatment ---
        magnitudes = np.array([mag for mag, dist in treatment_interactions])
        distances = np.array([dist for mag, dist in treatment_interactions])
        
        # Find the 99th percentile cutoff
        cutoff = np.percentile(magnitudes, 99)
        
        # Create masks
        top_1_mask = magnitudes >= cutoff
        remaining_mask = magnitudes < cutoff
        
        # Get the two distance distributions
        top_1_distances = distances[top_1_mask]
        remaining_distances = distances[remaining_mask]
        
        # --- NEW: Perform Mann-Whitney U test ---
        p_str = "n/a"
        if len(top_1_distances) > 1 and len(remaining_distances) > 1:
            try:
                # Use 'two-sided' as we don't have a prior hypothesis
                stat, pvalue = stats.mannwhitneyu(top_1_distances, 
                                                  remaining_distances, 
                                                  alternative='two-sided')
                # Format the p-value
                # --- MODIFICATION: Use significance stars ---
                if pvalue < 0.001:
                    p_str = "***"
                elif pvalue < 0.01:
                    p_str = "**"
                elif pvalue < 0.05:
                    p_str = "*"
                else:
                    p_str = "n.s."
                # --- END MODIFICATION ---
            except ValueError as e:
                # This can happen if all values are identical (no variance)
                print(f"Warning: Stat test failed for {treatment_name}: {e}")
                p_str = "test failed"
        else:
            p_str = "n.s." # Not enough samples to test
        
        p_values[treatment_name] = p_str
        # --- End of new section ---

        # Add data for plotting
        for d in top_1_distances:
            all_plot_data.append({
                "distance": d,
                "group": "Top 1%",
                "treatment": treatment_name
            })
            
        for d in remaining_distances:
            all_plot_data.append({
                "distance": d,
                "group": "Remaining 99%",
                "treatment": treatment_name
            })

    # Convert all data to a DataFrame for plotting
    if not all_plot_data:
        print("Error: No data was generated. Cannot create plot.")
        return
        
    df = pd.DataFrame(all_plot_data)
    
    # --- NEW: Get sorted order for treatments ---
    unique_treatments = df['treatment'].unique()
    # Sort treatments by PI count, then alphabetically for ties
    sorted_treatments = sorted(unique_treatments, key=lambda x: (get_pi_count(x), x))
    # --- End of new section ---
    
    print("\nGenerating plot...")
    
    # --- PLOTTING SECTION MODIFIED ---
    
    # Set a context for larger fonts, but we'll override for more control
    sns.set_context("paper", font_scale=1.5) 
    
    # Create the box plot
    fig, ax = plt.subplots(figsize=(10 + 2 * len(model_files), 9))
    sns.boxplot(
        ax=ax,
        data=df,
        x='treatment',
        y='distance',
        hue='group',
        order=sorted_treatments,
        # Use requested colors
        palette={"Top 1%": "salmon", "Remaining 99%": "lightblue"},
        showfliers=False
    )
    
    # Increase font sizes for readability
    ax.set_title("Spatial Distance Distribution of Top 1% vs. Remaining 99% Interactions", fontsize=22, weight='bold')
    ax.set_xlabel(None) # MODIFICATION: Remove x-axis label
    ax.set_ylabel("Cα-Cα Distance (Å)", fontsize=36) # Increased Y label font
    
    # --- NEW: Create wrapped labels for X-axis ---
    formatted_labels = []
    for name in sorted_treatments:
        count = get_pi_count(name)
        if count > 2:
            parts = name.split(',')
            # Find the middle comma to break on
            break_point = int(np.ceil(count / 2))
            
            line1 = ",".join(parts[:break_point])
            line2 = ",".join(parts[break_point:])
            formatted_labels.append(f"{line1},\n{line2}")
        else:
            formatted_labels.append(name)
    
    ax.set_xticklabels(formatted_labels, fontsize=32, ha='center')
    # --- END NEW SECTION ---

    # Increase tick label size (Y-axis only now)
    ax.tick_params(axis='y', which='major', labelsize=24)
    
    # Increase legend size and set location
    ax.legend(title="Interaction Group", fontsize=26, title_fontsize=20, loc='upper right', ncol=2) # MODIFICATION: Set location
    
    # --- MODIFICATION: Add stars *and brackets* above the plots ---
    
    # Calculate y-offset for the stars
    # Find the maximum y-value of any whisker in the plot
    global_max_y = df['distance'].max()
    # --- MODIFICATION: Lowered the offset ---
    # Set the offset to be 2% of the total y-range
    y_offset = global_max_y * 0.02 
    # --- END MODIFICATION ---
    # Set the height for the bracket's vertical ticks
    y_tick_height = y_offset * 0.4

    # Get current x-tick positions (0, 1, 2, ...)
    xticks = ax.get_xticks()
    
    # --- MODIFICATION: Calculate max whisker position correctly ---
    # Find the max y-value for *each* treatment group's WHISKER
    treatment_maxes = {}
    for treatment_name in sorted_treatments:
        # Get max whisker value for this treatment (max of either group)
        # Since showfliers=False, .max() is the top of the whisker
        try:
            treatment_data = df[df['treatment'] == treatment_name]
            
            # Calculate upper whisker for 'Top 1%'
            top_1_data = treatment_data[treatment_data['group'] == 'Top 1%']['distance']
            Q3_top = top_1_data.quantile(0.75)
            Q1_top = top_1_data.quantile(0.25)
            IQR_top = Q3_top - Q1_top
            limit_top = Q3_top + 1.5 * IQR_top
            whisker_top = top_1_data[top_1_data <= limit_top].max()
            
            # Calculate upper whisker for 'Remaining 99%'
            rem_data = treatment_data[treatment_data['group'] == 'Remaining 99%']['distance']
            Q3_rem = rem_data.quantile(0.75)
            Q1_rem = rem_data.quantile(0.25)
            IQR_rem = Q3_rem - Q1_rem
            limit_rem = Q3_rem + 1.5 * IQR_rem
            whisker_rem = rem_data[rem_data <= limit_rem].max()
            
            treatment_maxes[treatment_name] = max(whisker_top, whisker_rem)
        except Exception as e:
            # Handle empty data slices, e.g., if a treatment has no 'Top 1%'
             print(f"Warning: Could not calculate whisker for {treatment_name}. Error: {e}")
             if not np.isnan(whisker_top):
                 treatment_maxes[treatment_name] = whisker_top
             elif not np.isnan(whisker_rem):
                 treatment_maxes[treatment_name] = whisker_rem
             else:
                 treatment_maxes[treatment_name] = 0 # Fallback
                 
    # Find the global max whisker height for plot padding
    global_max_whisker = max(treatment_maxes.values())
    
    # Set the offset to be 2% of this max whisker height
    y_offset = global_max_whisker * 0.02 
    # Set the height for the bracket's vertical ticks
    y_tick_height = y_offset * 0.4
    # --- END MODIFICATION ---

    # Iterate over our sorted_treatments list to guarantee correct order
    for i, treatment_name in enumerate(sorted_treatments):
        p_str = p_values.get(treatment_name, "n.s.")
        
        # Get the y position for the bracket's horizontal bar
        # Place it above the tallest whisker for that group
        y_bracket_top = treatment_maxes.get(treatment_name, 0) + y_offset
        y_bracket_bottom = y_bracket_top - y_tick_height
        
        # Get the x positions for the two boxes
        # Approximate width of the dodged boxes
        box_width = 0.2
        x1 = xticks[i] - box_width
        x2 = xticks[i] + box_width
        
        # Plot the bracket lines
        if p_str != "n.s.":
            ax.plot([x1, x1, x2, x2], 
                    [y_bracket_bottom, y_bracket_top, y_bracket_top, y_bracket_bottom], 
                    color='black', 
                    linewidth=1.5)
        
            # Get the y position for the star (just above the bracket)
            y_star = y_bracket_top + (y_offset * 0.1)
        
            # Add the text
            ax.text(xticks[i], y_star, p_str, 
                    ha='center', # Horizontal alignment
                    va='bottom', # Vertical alignment
                    size='x-large', 
                    color='black',
                    weight='bold') 
    
    # Adjust plot Y-limit to make space for the highest star
    current_ylim = ax.get_ylim()
    # Find the position of the highest star
    try:
        highest_bracket_pos = max(treatment_maxes.values()) + y_offset
        highest_star_pos = highest_bracket_pos + (y_offset * 0.1)
        # --- MODIFICATION: Increased top padding for legend ---
        # Set new top limit, adding 20% padding above the highest star
        new_top_lim = highest_star_pos + (global_max_whisker * 0.20)
        # --- END MODIFICATION ---
    except ValueError:
        # Handle case where treatment_maxes is empty
        new_top_lim = current_ylim[1] # Keep original limit
        
    ax.set_ylim(bottom=current_ylim[0], top=new_top_lim)
    
    # --- END OF MODIFICATION ---
    
    # Use tight_layout to reduce whitespace
    # pad=2.0 adds a bit of padding so text isn't cramped
    plt.tight_layout(pad=2.0) 
    
    output_plot_file = "interaction_distance_boxplot.png"
    # Use bbox_inches='tight' to further trim whitespace upon saving
    plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
    
    # --- END OF PLOTTING SECTION ---
    
    print(f"\nSuccessfully saved plot to: {os.path.abspath(output_plot_file)}")

if __name__ == "__main__":
    # Configuration
    DIST_MATRIX_FILE = "hiv_protease_1HHP_A_dist_matrix.npy"
    MODEL_DIR = "../../train_models/models/" # ASSUMES your .p files are in a 'models' subfolder
    
    # Ensure you have pandas, seaborn, matplotlib, tqdm, and scipy installed
    # pip install pandas seaborn matplotlib tqdm scipy
    
    analyze_interactions(DIST_MATRIX_FILE, MODEL_DIR)
