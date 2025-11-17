import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

# --- 1. Load and Aggregate Data ---

fname = '../../simulate_trajectories/consolidated_binned_phi_1000.p'
dbfile = open(fname, 'rb')
res = pickle.load(dbfile)
dbfile.close()

all_experiment_phi = dict()

for rr in res:
    treatment, beta, LOO, experiment_phi = rr

    # Only aggregate data for LOO == -1, as it's the only one used
    if LOO != -1:
        continue

    if treatment not in all_experiment_phi:
        all_experiment_phi[treatment] = dict()

    for drug in experiment_phi:
        if drug not in all_experiment_phi[treatment]:
            all_experiment_phi[treatment][drug] = experiment_phi[drug]
        else:
            # Simply add counts in the bins across different beta
            all_experiment_phi[treatment][drug] += experiment_phi[drug]


# Define consistent line styles
treatment_linestyles = {
    'None': '--',  # Dashed for None (though we don't plot it)
    # All others get solid lines (default)
}

# Define X-axis
xx = np.linspace(0, math.sqrt(2), endpoint=False, num=1000)

# --- 2. Prepare Data for Plotting ---
all_plot_data = {}
do_drugs = ['NFV', 'ATV', 'SQV']

# Collect all cumulative sum data
for drug in do_drugs:
    all_plot_data[drug] = {}
    
    # Skip this drug if the 'None' (baseline) treatment is missing
    if 'None' not in all_experiment_phi or drug not in all_experiment_phi['None']:
        print(f"Warning: 'None' treatment or drug {drug} missing. Skipping drug.")
        continue

    for treatment in all_experiment_phi:
        # Skip if drug data is missing for this treatment
        if drug not in all_experiment_phi[treatment]:
            continue
            
        treatment_cumsum = np.cumsum(all_experiment_phi[treatment][drug])
        treatment_final_val = treatment_cumsum[-1]
        
        # Skip if final cumulative value is less than 10
        if treatment_final_val < 10:
            continue
            
        all_plot_data[drug][treatment] = treatment_cumsum

# --- 3. Generate Plots ---
eps = 10

for drug in do_drugs:
    
    treatment_data = all_plot_data.get(drug)
    
    # Skip if no valid data was found, or if the 'None' baseline is missing
    if not treatment_data or 'None' not in treatment_data:
        print(f"No valid data or 'None' baseline for {drug}, skipping plot")
        continue

    # Create a new figure for each drug
    plt.figure(figsize=(10, 8))
    
    # --- New Legend Logic ---
    mono_therapy_labeled = False
    multi_therapy_labeled = False
    #ATV_RTV = False
    
    # Pre-calculate the denominator (baseline)
    none_data = treatment_data['None'] + eps

    for treatment, data in treatment_data.items():
        linestyle = treatment_linestyles.get(treatment, '-')
        
        n_PIs = len(treatment.split(','))
        label_to_use = None # Default to no label

        if treatment == 'None':
            # Do not plot the 'None' line, it's the baseline denominator
            continue
        elif n_PIs == 1:
            color = 'blue'
            if not mono_therapy_labeled:
                label_to_use = 'Mono-PI treat. regimen'
                mono_therapy_labeled = True
        else: # n_PIs > 1
            color = 'red'
            #if not multi_therapy_labeled and treatment != 'ATV,RTV':
            if not multi_therapy_labeled:
                label_to_use = 'Multi-PI treat. regimen'
                multi_therapy_labeled = True


        #if treatment == 'ATV,RTV':
        #    alpha=1.0
        #    lw=2
        #    ATV_RTV = True
        #    label_to_use = 'ATV,RTV treat. regimen'
        #else:
        #    alpha=0.8
        #    lw=1

        
        # Calculate the ratio
        y_values = (data + eps) / none_data
        
        plt.plot(xx, y_values, 
                 label=label_to_use, 
                 color=color,
                 linestyle=linestyle,
                 alpha=1.0,
                 linewidth=2)

    plt.yscale('log')
    plt.title(f'{drug} resistant genotypes', fontsize=28)
    plt.xlabel(r'$\tau$', fontsize=30)

    # --- Y-Label Correction ---
    # Removed r'' prefix so \n is interpreted as a newline.
    # Using string concatenation to safely combine text and newline.
    plt.ylabel(r'Fractional change in $\phi(\tau)$' + '\nwrt to no treatment', fontsize=30)

    plt.gca().tick_params(axis='x', which='major', labelsize=22)
    plt.gca().tick_params(axis='y', which='major', labelsize=22)
    
    # --- Add Legend ---
    plt.legend(fontsize=20)
    
    plt.tight_layout()
    plt.savefig(f'fig4b_drug{drug}.jpg', dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory before the next loop




