import math
import matplotlib.pyplot as plt
import numpy as np
import pickle



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
            # Simply add counts in the bins across different beta
            all_experiment_phi[treatment][LOO][drug] += experiment_phi[drug]




# Get all unique treatments first
all_treatments = set()
for treatment in all_experiment_phi:
    all_treatments.add(treatment)

# Define a consistent color mapping for ALL treatments
colors = ['black', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
          'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
          'darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet',
          'goldenrod', 'indigo', 'crimson', 'forestgreen', 'navy']

# Create mapping ensuring 'None' gets black and is dashed
treatment_colors = {}
sorted_treatments = sorted(all_treatments)

# Assign 'None' first if it exists
if 'None' in sorted_treatments:
    treatment_colors['None'] = 'black'
    sorted_treatments.remove('None')

# Assign colors to remaining treatments
for i, treatment in enumerate(sorted_treatments):
    treatment_colors[treatment] = colors[(i + 1) % len(colors)]  # +1 to skip black

# Define consistent line styles
treatment_linestyles = {
    'None': '--',  # Dashed for None
    # All others get solid lines (default)
}

xx = np.linspace(0, math.sqrt(2), endpoint=False, num=1000)

# Pre-calculate all data to determine global axis limits
all_plot_data = {}
global_y_min = float('inf')
global_y_max = float('-inf')

do_drugs = ['NFV', 'ATV', 'SQV']

# First pass: collect all data and find global limits
# for drug in all_experiment_phi['None'][-1]:

for drug in do_drugs:
    all_plot_data[drug] = {}
    
    for treatment in all_experiment_phi:
        treatment_cumsum = np.cumsum(all_experiment_phi[treatment][-1][drug])
        treatment_final_val = treatment_cumsum[-1]
        
        # Skip if final cumulative value is less than 10
        if treatment_final_val < 10:
            continue
            
        all_plot_data[drug][treatment] = treatment_cumsum
        
        # Update global limits
        y_min, y_max = np.min(treatment_cumsum), np.max(treatment_cumsum)
        global_y_min = min(global_y_min, y_min)
        global_y_max = max(global_y_max, y_max)

# Add some padding to the limits (for log scale, use multiplicative padding)
global_y_min = max(1, global_y_min * 0.5)  # Don't go below 1 for log scale
global_y_max *= 2.0

# Second pass: create plots with consistent limits
# for drug in all_experiment_phi['None'][-1]:


do_treatments = ['IDV,RTV,SQV', 'None']
do_drugs = ['NFV']

eps = 10

for drug in do_drugs:

    fig, axs = plt.subplots(1, 2, figsize=(10,6))

    # Use pre-calculated data for this drug
    treatment_data = all_plot_data[drug]
    
    if not treatment_data:  # Skip if no valid treatments for this drug
        print(f"No valid treatments for {drug}, skipping plot")
        plt.close()
        continue
    
    for treatment in do_treatments:
        color = treatment_colors[treatment]
        linestyle = treatment_linestyles.get(treatment, '-')  # Default to solid line

        #n_PIs = len(treatment.split(','))
        #if treatment == 'None':
        #    color='black'
        #elif n_PIs == 1:
        #    color='lightblue'
        #else:
        #    color='salmon'
        
        axs[0].plot(xx, treatment_data[treatment], 
                label=treatment, 
                color=color,
                linestyle=linestyle,
                linewidth=2)

        axs[0].set_yscale('log')

        if treatment != 'None':

            axs[1].plot(xx, (treatment_data[treatment]+eps)/(treatment_data['None']+eps), 
                label=treatment, 
                color=color,
                linestyle=linestyle,
                linewidth=2)

            axs[1].set_yscale('log')
    
    plt.suptitle(f'{drug} resistant genotypes', fontsize=24)

    axs[0].legend(fontsize=16)
    axs[1].legend(fontsize=16)

    axs[0].set_xlabel(r'$\tau$', fontsize=22)
    axs[0].set_ylabel(r'$\phi(\tau)$', fontsize=22)

    axs[1].set_xlabel(r'$\tau$', fontsize=22)
    axs[1].set_ylabel(r'Fractional change in $\phi(\tau)$' + '\nwrt to no treatment', fontsize=22)
    

    # Increase tick label font sizes
    axs[0].tick_params(axis='x', which='major', labelsize=16)
    axs[0].tick_params(axis='y', which='major', labelsize=16)

    axs[1].tick_params(axis='x', which='major', labelsize=16)
    axs[1].tick_params(axis='y', which='major', labelsize=16)

    #plt.xlabel(r'$\tau$', fontsize=24)
    #plt.ylabel(r'$\phi(\tau)$', fontsize=24)
    #plt.yscale('log')

    #
    ## Increase tick label font sizes
    #plt.tick_params(axis='x', which='major', labelsize=14)
    #plt.tick_params(axis='y', which='major', labelsize=14)
    #
    ## Set consistent axis limits across all plots
    #plt.xlim(0, math.sqrt(2))
    #plt.ylim(global_y_min, global_y_max)
    #
    ## Sort legend entries for consistency
    #handles, labels = plt.gca().get_legend_handles_labels()
    #sorted_pairs = sorted(zip(labels, handles))
    #sorted_labels, sorted_handles = zip(*sorted_pairs)
    #
    #plt.legend(sorted_handles, sorted_labels, fontsize=16)
    plt.tight_layout()

    ##plt.savefig(f'fig4a_drug{drug}_revision.jpg', dpi=300, bbox_inches='tight')

    ##plt.cla()
    ##plt.clf()
    ##plt.close()
    
    #plt.show()
    #exit()
    plt.savefig(f'fig4a_drug{drug}.jpg', dpi=300, bbox_inches='tight')

    exit()

