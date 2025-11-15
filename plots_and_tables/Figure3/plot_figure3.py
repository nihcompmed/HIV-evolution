import pickle
import matplotlib.pyplot as plt
import glob
from joblib import Parallel, delayed
import numpy as np
import matplotlib as mpl
from scipy.interpolate import griddata


# PLOT Figure 3a to show scatter plot of log(-log prog to reach drug res) and mutation propensity

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


from tqdm import tqdm
# Figure 3 to show scatter plot of log(-log prog to reach drug res) and mutation propensity

# Proposal: log(-log prog to reach drug res) -> 'Drug Susceptibility'

example_treatment = 'IDV,RTV,SQV'
example_drug_res = 'NFV'

fname = f'all_unique_stats_treatment{example_treatment}_drug{example_drug_res}.npz'
data = np.load(fname)

all_unique_fitness = data['all_unique_fitness']
all_unique_log_log_prob = data['all_unique_log_log_prob']


fig, axs = plt.subplots(1, 3, figsize=(18,6))

#axs[0].scatter(all_unique_log_log_prob, all_unique_fitness, alpha=0.4, color='blue')
#axs[0].set_xlabel(f'1/reachability', fontsize=22)
#axs[0].set_ylabel(f'Mutation propensity', fontsize=22)

# Store the return value from hist2d
im = axs[0].hist2d(all_unique_log_log_prob, all_unique_fitness, norm=mpl.colors.LogNorm(), cmap=mpl.cm.gnuplot, bins=50)

# Use fig.colorbar() and pass the mappable object
cbar = fig.colorbar(im[3], ax=axs[0])  # im[3] is the mappable object
cbar.set_label('#genotypes', fontsize=14)

axs[0].set_xlabel(f'1/reachability', fontsize=18)
axs[0].set_ylabel(f'Mutation propensity', fontsize=18)


normalization_factor_prob_to_reach = 10
normalization_mutation_propensity = 20

scaled_prob_to_reach = all_unique_log_log_prob/normalization_factor_prob_to_reach
scaled_fitness = all_unique_fitness/normalization_mutation_propensity
phi_norm = np.sqrt(scaled_prob_to_reach**2 + scaled_fitness**2)

plot_phi_thresh = 0.45
x = scaled_prob_to_reach
y = scaled_fitness
z = phi_norm
row = 0
col = 0
marker_points = []

# Create color array based on threshold
colors = ['salmon' if z_val < plot_phi_thresh else 'lightblue' for z_val in z]

# Count points below threshold
count_below_thresh = sum(1 for z_val in z if z_val < plot_phi_thresh)

# Create scatter plot
axs[1].scatter(x, y, c=colors, alpha=0.8)
axs[1].set_xlabel(f'1/reachability\n(normalized)', fontsize=18)
axs[1].set_ylabel(f'Mutation propensity (normalized)', fontsize=18)

#axs[col].set_title(f'Pareto optimality threshold {thresh}\ncumulative counts below threshold {count_below_thresh}', fontsize=20)

# Increase tick label font sizes
axs[1].tick_params(axis='both', which='major', labelsize=12)


# Plot cumulative counts
values_sorted = np.sort(phi_norm)
cumulative_counts = np.arange(1, len(values_sorted) + 1)
axs[2].plot(values_sorted, cumulative_counts, linewidth=2)

#for pt in marker_points:
#    axs[1].scatter(pt[0], pt[1], color='red', zorder=1000, s=50)
#    axs[1].text(pt[0], pt[1], f'({pt[0]},{pt[1]})', fontsize=20, zorder=1000, 
#                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.2))

axs[2].set_yscale('log')
axs[2].set_xlabel(r'$\tau$', fontsize=22)
axs[2].set_ylabel(r'$\phi(\tau)$', fontsize=22)
axs[2].tick_params(axis='both', which='major', labelsize=12)

x_scatter = x
y_scatter = y
z_scatter = z


# Define the grid resolution
grid_res = 100
# Create the 1D x and y coordinates for the grid
grid_x_1d = np.linspace(x_scatter.min(), x_scatter.max(), grid_res)
grid_y_1d = np.linspace(y_scatter.min(), y_scatter.max(), grid_res)
# Create the 2D grid coordinates
grid_x_2d, grid_y_2d = np.meshgrid(grid_x_1d, grid_y_1d)

# Interpolate the z-values onto the 2D grid
# 'cubic' interpolation is smoother, 'linear' is faster
grid_z_2d = griddata(
    (x_scatter, y_scatter), z_scatter, (grid_x_2d, grid_y_2d), method='cubic'
)

# Define the *single* level you want to plot
level_to_plot = [0.45]

# Generate the contour plot
# We only pass the single level to the 'levels' argument
cs = axs[1].contour(
    grid_x_2d,
    grid_y_2d,
    grid_z_2d,
    levels=level_to_plot,
    colors='black',  # Specify a color for the line
    linewidths=2
)

# --- 4. Annotate the Contour Line ---
# To annotate with a specific string (not the numeric value),
# we pass a dictionary to the 'fmt' (format) argument of clabel.
# The dictionary maps the level value (0.45) to the desired string.
label_format = {0.45: r'$\tau=0.45$'}  # Use a raw string (r'...') for LaTeX

axs[1].clabel(
    cs,
    inline=True,      # Place label inline with the contour
    fontsize=24,
    fmt=label_format, # Use our custom format
)




# Add main title with proper spacing
plt.suptitle(f'{example_drug_res} resistant genotypes in treatment regimen {example_treatment}', fontsize=22)

# Apply tight layout with padding for the suptitle and extra vertical spacing
#plt.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3.0)
plt.tight_layout()

plt.savefig(f'fig3.jpg', dpi=300, bbox_inches='tight')
plt.show()


