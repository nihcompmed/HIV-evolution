import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

# --- 1. Load and Aggregate Data ---
print("Loading data...")

# Get onehot_inv_maps of all treatments
fname = '../../preprocess_data/PI_treatments_dict.p'
with open(fname, 'rb') as dbfile:
    PI_treatments = pickle.load(dbfile)

model_dirr = '../../train_models/models'
onehot_inv_map = dict()

for treatment in PI_treatments:
    fname = f'{model_dirr}/treatment{treatment}_PR_evol_onehot_logistic.p'
    try:
        with open(fname, 'rb') as dbfile:
            info = pickle.load(dbfile)
    except FileNotFoundError:
        # print(f"Warning: Could not find model file for {treatment}, skipping.")
        continue
    
    PR_onehot_encoder = info['onehot_encoder']
    onehot_inv_map[treatment] = dict()
    bit_idx = 0 
    for pos, cat in enumerate(PR_onehot_encoder.categories_):
        for AA in cat:
            onehot_inv_map[treatment][bit_idx] = (pos, AA)
            bit_idx += 1

# Load main simulation data
fname = '../../simulate_trajectories/consolidated_binned_phi_1000.p'
dbfile = open(fname, 'rb')
res = pickle.load(dbfile)
dbfile.close()

all_experiment_phi = dict()

for rr in res:
    treatment, beta, LOO, experiment_phi = rr
    if treatment not in all_experiment_phi:
        all_experiment_phi[treatment] = dict()
    if LOO not in all_experiment_phi[treatment]:
        all_experiment_phi[treatment][LOO] = dict()
    for drug in experiment_phi:
        if drug not in all_experiment_phi[treatment][LOO]:
            all_experiment_phi[treatment][LOO][drug] = experiment_phi[drug]
        else:
            all_experiment_phi[treatment][LOO][drug] += experiment_phi[drug]

print("Data loading complete.")

# --- 2. Setup Plotting Grid ---

# Define X-axis
xx = np.linspace(0, math.sqrt(2), endpoint=False, num=1000)
eps = 10

# Define the grid structure
do_drugs = ['NFV', 'ATV', 'SQV']
do_treatments = ['ATV,RTV']
# Format treatment names for titles (original names for titles)
treatment_titles = [t.replace(",", ", ") for t in do_treatments]

N_rows = len(do_treatments)
N_cols = len(do_drugs)

# Colors for top 3 LOOs
highlight_colors = ['red', 'blue', 'green']

# Create the master figure and subplot grid
fig, axes = plt.subplots(N_rows, N_cols, figsize=(20,6),
                         sharex=True, sharey=True)

#fig.suptitle('LOO Impact Analysis by Drug and Treatment Regimen', fontsize=40, y=1.03)


# --- 3. Iterate over Grid and Plot ---

print("Generating subplot grid...")

for col_idx, drug in enumerate(do_drugs):
    
    # Get the specific axis for this subplot
    ax = axes[col_idx]

    # Check if data exists
    if treatment not in all_experiment_phi or treatment == 'None' or treatment not in onehot_inv_map:
        print(f"Skipping {treatment}::{drug} - Missing treatment data.")
        ax.axis('off')
        continue
        
    LOO_min_change = []
    LOO_list = []
    valid_plots = []  # Store valid fractional change data
    valid_LOOs = []   # Store corresponding LOO indices
    
    # Check for baseline data
    if -1 not in all_experiment_phi[treatment] or drug not in all_experiment_phi[treatment][-1]:
        print(f"Skipping {treatment}::{drug} - Missing baseline (-1) data.")
        ax.axis('off')
        continue

    val2_cumsum = np.cumsum(all_experiment_phi[treatment][-1][drug])

    for LOO in all_experiment_phi[treatment]:
        if LOO == -1:
            continue
        
        # Check for this LOO's data
        if drug not in all_experiment_phi[treatment][LOO]:
            continue
            
        val1_cumsum = np.cumsum(all_experiment_phi[treatment][LOO][drug])
        
        if val1_cumsum[-1] < 10 and val2_cumsum[-1] < 10:
            continue
        
        frac_change = (eps + val1_cumsum) / (eps + val2_cumsum)
        
        mask = (xx >= 0.2) & (xx <= 0.6)
        xx_subset = xx[mask]
        frac_change_subset = frac_change[mask]
        
        log_deviation = np.log(frac_change_subset)
        signed_log_area = np.trapz(log_deviation, xx_subset)
        
        LOO_min_change.append(-signed_log_area) # Store negative
        LOO_list.append(LOO)
        valid_plots.append(frac_change)
        valid_LOOs.append(LOO)
    
    if not valid_plots:
        print(f"No valid LOO data for {treatment}::{drug}, turning off subplot.")
        ax.axis('off')
        continue
    
    LOO_min_change = np.array(LOO_min_change)
    LOO_list = np.array(LOO_list)
    
    sorted_indices = np.argsort(LOO_min_change)[::-1]
    bot_n_indices = sorted_indices[:3] if len(sorted_indices) >= 3 else sorted_indices
    
    legend_info = []
    
    for i, frac_change in enumerate(valid_plots):
        ax.plot(xx, frac_change, color='gray', alpha=0.3, linewidth=1, zorder=1)
    
    for rank, original_idx in enumerate(bot_n_indices):
        frac_change = valid_plots[original_idx]
        current_LOO = valid_LOOs[original_idx]
        
        pos, AA = onehot_inv_map[treatment][current_LOO]
        zorder_val = 1000 - rank * 100
        
        line = ax.plot(xx, frac_change, 
                       color=highlight_colors[rank], 
                       alpha=0.8, 
                       linewidth=2,
                       zorder=zorder_val)[0]
        
        actual_signed_log_area = -LOO_min_change[original_idx]
        legend_info.append((actual_signed_log_area, f'{pos+1}{AA}', line))

    # --- Set Subplot-Specific Labels and Properties ---
    
    ax.set_yscale('log')
    ax.set_xlim(0, 0.8)
    
    # --- TICK FONTSIZE CHANGE ---
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    
    legend_info.sort(key=lambda x: x[0])
    handles = [info[2] for info in legend_info]
    labels = [info[1] for info in legend_info]
    ax.legend(handles, labels, fontsize=20, loc='best')
    
    # --- Set Grid-Specific Labels ---
    
    ax.set_title(f'Resist {drug}', fontsize=36)
    ax.set_ylabel(r'Fractional change in $\phi(\tau)$'+'\nwrt no LOO', fontsize=24)
    ax.set_xlabel(r'$\tau$', fontsize=30)
        
            
print("Grid generation complete.")

# --- 4. Finalize and Save ---

## Add a single, common Y-label for the entire figure (no line break)
#common_ylabel = r'Fractional change in $\phi(\tau)$ wrt no LOO'
## Place it centered, to the left of the subplots, with larger font
## --- GAP REDUCTION: Increased x from 0.06 to 0.08 ---
#fig.supylabel(common_ylabel, fontsize=52, x=0.08) 

# Adjust rect to make room for suptitle and new supylabel
# --- GAP REDUCTION: Decreased left margin from 0.1 to 0.09 ---
plt.tight_layout(rect=[0.09, 0.03, 1, 0.95]) 
plt.savefig('fig6A_LOO_grid.jpg', dpi=300, bbox_inches='tight')
plt.close()



# Create the master figure and subplot grid
fig, axes = plt.subplots(N_rows, N_cols, figsize=(20,6),
                         sharex=True, sharey=True)

#fig.suptitle('LOO Impact Analysis by Drug and Treatment Regimen', fontsize=40, y=1.03)


# --- 3. Iterate over Grid and Plot ---

print("Generating subplot grid...")

for col_idx, drug in enumerate(do_drugs):
    
    # Get the specific axis for this subplot
    ax = axes[col_idx]

    # Check if data exists
    if treatment not in all_experiment_phi or treatment == 'None' or treatment not in onehot_inv_map:
        print(f"Skipping {treatment}::{drug} - Missing treatment data.")
        ax.axis('off')
        continue
        
    LOO_min_change = []
    LOO_list = []
    valid_plots = []  # Store valid fractional change data
    valid_LOOs = []   # Store corresponding LOO indices
    
    # Check for baseline data
    if -1 not in all_experiment_phi[treatment] or drug not in all_experiment_phi[treatment][-1]:
        print(f"Skipping {treatment}::{drug} - Missing baseline (-1) data.")
        ax.axis('off')
        continue

    val2_cumsum = np.cumsum(all_experiment_phi[treatment][-1][drug])

    for LOO in all_experiment_phi[treatment]:
        if LOO == -1:
            continue
        
        # Check for this LOO's data
        if drug not in all_experiment_phi[treatment][LOO]:
            continue
            
        val1_cumsum = np.cumsum(all_experiment_phi[treatment][LOO][drug])
        
        if val1_cumsum[-1] < 10 and val2_cumsum[-1] < 10:
            continue
        
        frac_change = (eps + val1_cumsum) / (eps + val2_cumsum)
        
        mask = (xx >= 0.2) & (xx <= 0.6)
        xx_subset = xx[mask]
        frac_change_subset = frac_change[mask]
        
        log_deviation = np.log(frac_change_subset)
        signed_log_area = np.trapz(log_deviation, xx_subset)
        
        LOO_min_change.append(signed_log_area) # Store positive
        LOO_list.append(LOO)
        valid_plots.append(frac_change)
        valid_LOOs.append(LOO)
    
    if not valid_plots:
        print(f"No valid LOO data for {treatment}::{drug}, turning off subplot.")
        ax.axis('off')
        continue
    
    LOO_min_change = np.array(LOO_min_change)
    LOO_list = np.array(LOO_list)
    
    sorted_indices = np.argsort(LOO_min_change)[::-1]
    bot_n_indices = sorted_indices[:3] if len(sorted_indices) >= 3 else sorted_indices
    
    legend_info = []
    
    for i, frac_change in enumerate(valid_plots):
        ax.plot(xx, frac_change, color='gray', alpha=0.3, linewidth=1, zorder=1)
    
    for rank, original_idx in enumerate(bot_n_indices):
        frac_change = valid_plots[original_idx]
        current_LOO = valid_LOOs[original_idx]
        
        pos, AA = onehot_inv_map[treatment][current_LOO]
        zorder_val = 1000 - rank * 100
        
        line = ax.plot(xx, frac_change, 
                       color=highlight_colors[rank], 
                       alpha=0.8, 
                       linewidth=2,
                       zorder=zorder_val)[0]
        
        actual_signed_log_area = -LOO_min_change[original_idx]
        legend_info.append((actual_signed_log_area, f'{pos+1}{AA}', line))

    # --- Set Subplot-Specific Labels and Properties ---
    
    ax.set_yscale('log')
    ax.set_xlim(0, 0.8)
    
    # --- TICK FONTSIZE CHANGE ---
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=20)
    
    legend_info.sort(key=lambda x: x[0])
    handles = [info[2] for info in legend_info]
    labels = [info[1] for info in legend_info]
    ax.legend(handles, labels, fontsize=20, loc='best')
    
    # --- Set Grid-Specific Labels ---
    
    ax.set_title(f'Resist {drug}', fontsize=36)

    ax.set_ylabel(r'Fractional change in $\phi(\tau)$'+'\nwrt no LOO', fontsize=24)
    ax.set_xlabel(r'$\tau$', fontsize=30)
    
        
            
print("Grid generation complete.")

# --- 4. Finalize and Save ---

## Add a single, common Y-label for the entire figure (no line break)
#common_ylabel = r'Fractional change in $\phi(\tau)$ wrt no LOO'
## Place it centered, to the left of the subplots, with larger font
## --- GAP REDUCTION: Increased x from 0.06 to 0.08 ---
#fig.supylabel(common_ylabel, fontsize=52, x=0.08) 

# Adjust rect to make room for suptitle and new supylabel
# --- GAP REDUCTION: Decreased left margin from 0.1 to 0.09 ---
plt.tight_layout(rect=[0.09, 0.03, 1, 0.95]) 
plt.savefig('fig6B_LOO_grid.jpg', dpi=300, bbox_inches='tight')
plt.close()





