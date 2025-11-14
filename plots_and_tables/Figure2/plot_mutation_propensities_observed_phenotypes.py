import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Load and plot mutation propensities of observed genotypes within each treatment

Creates boxplots showing distribution of mutation propensities for sequences
observed under each treatment condition
"""

print("\n" + "="*80)
print("PLOTTING MUTATION PROPENSITIES OF OBSERVED GENOTYPES")
print("="*80)

# Load results
print("\nLoading results...")
with open('mutation_propensity_observed_results.p', 'rb') as f:
    mutation_propensities = pickle.load(f)

print(f"Loaded results for {len(mutation_propensities.keys())} treatments")

treatments = list(mutation_propensities.keys())

# Print summary
print("\n" + "="*80)
print("SUMMARY STATISTICS (Unsorted)")
print("="*80)
print(f"\n{'Treatment':<20} {'N':<10} {'Mean':<10} {'Median':<10} {'Std':<10}")
print("-"*70)

for treatment in treatments:
    vals = mutation_propensities[treatment]
    print(f"{treatment:<20} {len(vals):<10} {np.mean(vals):<10.3f} "
          f"{np.median(vals):<10.3f} {np.std(vals):<10.3f}")

print("-"*70)

# --- MODIFICATION: Sort treatments by decreasing median for boxplot ---
print("\nSorting treatments by decreasing median propensity for boxplot...")

def get_pi_count(treatment_name):
    """Helper function for label wrapping."""
    if treatment_name == 'None':
        return 0
    # Check for ',' as the delimiter
    if ',' in treatment_name:
        return len(treatment_name.split(','))
    # Otherwise, it's a single PI
    return 1

# Sort treatments by decreasing median
sorted_treatments_by_median = sorted(treatments, key=lambda t: np.median(mutation_propensities[t]), reverse=True)
print(f"Sorted order: {', '.join(sorted_treatments_by_median)}")
# --- END MODIFICATION ---


# Create boxplot
print("\nCreating visualization (Boxplot)...")

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data -- MODIFIED to use median-sorted list
data_for_plot = [mutation_propensities[t] for t in sorted_treatments_by_median]

# Create boxplot
bp = ax.boxplot(data_for_plot, patch_artist=True,
                showfliers=False, # Hide outliers
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'linestyle': '--', 'linewidth': 1.0, 'color': 'black'},
                capprops={'linewidth': 1.0, 'color': 'black'},
                boxprops={'facecolor': 'lightblue', 'edgecolor': 'black', 'linewidth': 1.0})

ax.set_xlabel('Treatment regimen', fontsize=22)
ax.set_ylabel('Mutation propensity', fontsize=24)
ax.set_title('Distribution of mutation propensity for observed genotypes by treatment',
              fontsize=14, fontweight='bold')
# MODIFIED to use median-sorted list
ax.set_xticks(np.arange(1, len(sorted_treatments_by_median) + 1))

# --- NEW: Create wrapped labels for X-axis ---
formatted_labels = []
# MODIFIED to use median-sorted list
for name in sorted_treatments_by_median:
    count = get_pi_count(name)
    if count > 2:
        parts = name.split(',')
        # Break after the second PI
        line1 = ",".join(parts[:2])
        line2 = ",".join(parts[2:])
        formatted_labels.append(f"{line1},\n{line2}")
    else:
        formatted_labels.append(name)

# MODIFIED to use new formatted labels and remove rotation
ax.set_xticklabels(formatted_labels, ha='center', fontsize=18)
# --- END MODIFICATION ---

ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mutation_propensity_observed_genotypes.jpg', dpi=300, bbox_inches='tight')
print("Saved figure: mutation_propensity_observed_genotypes.jpg")
plt.close()

# Create bar plot (sorted by median)
print("\nCreating visualization (Bar plot by median)...")
# This section remains sorted by *increasing* median, as it was in the original script
treatments_sorted = sorted(treatments, key=lambda t: np.median(mutation_propensities[t]))
median_sorted_data = [mutation_propensities[t] for t in treatments_sorted]
means = [np.mean(d) for d in median_sorted_data]
stds = [np.std(d) for d in median_sorted_data]
positions = np.arange(len(treatments_sorted))

fig, ax3 = plt.subplots(figsize=(14, 8))
ax3.bar(positions, means, yerr=stds, align='center',
               alpha=0.7, color='cornflowerblue', edgecolor='black', linewidth=1.5,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 5})

ax3.set_xlabel('Treatment Regimen', fontsize=16, fontweight='bold')
ax3.set_ylabel('Mean Mutation Propensity ± Std', fontsize=16, fontweight='bold')
ax3.set_title('Mean Mutation Propensity of Observed Genotypes by Treatment (Sorted by Median)',
              fontsize=14, fontweight='bold')
ax3.set_xticks(positions)

# --- NEW: Apply same label wrapping logic to the bar plot ---
formatted_bar_labels = []
for name in treatments_sorted: # Note: uses the median-sorted list
    count = get_pi_count(name)
    if count > 2:
        parts = name.split(',')
        # Break after the second PI
        line1 = ",".join(parts[:2])
        line2 = ",".join(parts[2:])
        formatted_bar_labels.append(f"{line1},\n{line2}")
    else:
        formatted_bar_labels.append(name)
        
ax3.set_xticklabels(formatted_bar_labels, ha='center', fontsize=16) # Removed rotation
# --- END MODIFICATION ---

ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mutation_propensity_observed_means.jpg', dpi=300, bbox_inches='tight')
print("Saved figure: mutation_propensity_observed_means.jpg")
plt.close()

# Summary table (sorted by median)
print("\n" + "="*80)
print("TREATMENTS RANKED BY MEDIAN MUTATION PROPENSITY")
print("="*80)
print(f"\n{'Rank':<6} {'Treatment':<20} {'Median':<10} {'Mean':<10} {'Std':<10} {'N':<10}")
print("-"*70)

# This also remains sorted by *increasing* median, as in the original script
for rank, treatment in enumerate(treatments_sorted, 1):
    vals = mutation_propensities[treatment]
    print(f"{rank:<6} {treatment:<20} {np.median(vals):<10.3f} {np.mean(vals):<10.3f} "
          f"{np.std(vals):<10.3f} {len(vals):<10}")
print("-"*70)
print("\nDone.")
