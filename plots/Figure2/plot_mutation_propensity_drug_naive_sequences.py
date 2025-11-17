import pickle
import numpy as np
import matplotlib.pyplot as plt

"""
Load mutation propensity results and plot DIFFERENCE from 'None' treatment

Plots: Δm_F = m_F(σ) - m_None(σ)
Positive values = higher mutation propensity than in drug-naive landscape
"""

print("\n" + "="*80)
print("PLOTTING MUTATION PROPENSITY DIFFERENCES")
print("="*80)

# --- NEW: Helper function for label formatting ---
def get_pi_count(treatment_name):
    """Helper function to determine the sort order of treatments."""
    if treatment_name == 'None':
        return 0
    # Check for ',' as the delimiter
    if ',' in treatment_name:
        return len(treatment_name.split(','))
    # Otherwise, it's a single PI
    return 1
# --- END NEW ---

# Load results
print("\nLoading results...")
with open('mutation_propensity_drug_naive_results.p', 'rb') as f:
    results = pickle.load(f)

print(f"Loaded results for {len(results['treatments'])} treatments")
print(f"Sequences analyzed: {results['n_encodable']}")

treatments = results['treatments']
mutation_propensities = results['mutation_propensities']

# Check that 'None' is in the data
if 'None' not in treatments:
    print("ERROR: 'None' treatment not found in results!")
    exit(1)

# Get 'None' treatment propensities as baseline
baseline_propensities = mutation_propensities['None']
print(f"\nBaseline ('None') mutation propensities:")
print(f"  N: {len(baseline_propensities)}")
print(f"  Mean: {np.mean(baseline_propensities):.3f}")
print(f"  Median: {np.median(baseline_propensities):.3f}")
print(f"  Std: {np.std(baseline_propensities):.3f}")

# Compute differences from 'None' for all other treatments
print("\nComputing differences from baseline...")
differences = {}
other_treatments = [t for t in treatments if t != 'None']

for treatment in other_treatments:
    # Ensure propensities are available for this treatment
    if treatment in mutation_propensities:
        treatment_propensities = mutation_propensities[treatment]
        # Calculate difference (assuming sequences are aligned/matched)
        diff = np.array(treatment_propensities) - np.array(baseline_propensities)
        differences[treatment] = diff
        print(f"  {treatment:<20} | Median Δm: {np.median(diff):<8.3f} | Mean Δm: {np.mean(diff):<8.3f}")
    else:
        print(f"  Warning: No propensities found for treatment '{treatment}'. Skipping.")

if not differences:
    print("ERROR: No other treatments found to compare with 'None'. Exiting.")
    exit(1)

# --- Sort treatments by MEDIAN difference ---
print("\nSorting treatments by median difference...")
treatments_sorted = sorted(other_treatments, key=lambda t: np.median(differences[t]))

# --- MODIFICATION: Prepare data for boxplot ---
# Prepare list of arrays for boxplot
data_for_plot = [differences[t] for t in treatments_sorted]
# Boxplot uses 1-based indexing for positions
positions_sorted = np.arange(1, len(treatments_sorted) + 1)
# --- END MODIFICATION ---


# --- MODIFICATION: Plot boxplot instead of bar chart ---
print("\nCreating visualization (Boxplot)...")
fig, ax2 = plt.subplots(figsize=(14, 8))

# Replaced ax2.bar with ax2.boxplot
ax2.boxplot(data_for_plot, patch_artist=True,
            showfliers=False, # Hide outliers
            positions=positions_sorted, # Ensure positions match labels
            medianprops={'color': 'black', 'linewidth': 1.5},
            whiskerprops={'linestyle': '--', 'linewidth': 1.0, 'color': 'black'},
            capprops={'linewidth': 1.0, 'color': 'black'},
            boxprops={'facecolor': 'lightblue', 'edgecolor': 'black', 'linewidth': 1.0})
# --- END MODIFICATION ---

# Add baseline (zero difference)
ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='Baseline (No treatment)')

# Formatting with fontsize=16
#ax2.set_xlabel('Treatment regimen', fontsize=22)
ax2.set_ylabel(r'Change in mutation propensity: $\Delta m_F$',
               fontsize=24)
ax2.set_title(f'Mutation propensity change of drug-naive genotypes wrt to no treatment',
              fontsize=24)
ax2.set_xticks(positions_sorted)


# --- MODIFICATION: Create wrapped labels for X-axis ---
formatted_labels = []
for name in treatments_sorted: # Use the existing sorted list
    count = get_pi_count(name)
    if count > 2:
        parts = name.split(',')
        # Break after the second PI
        line1 = ",".join(parts[:2])
        line2 = ",".join(parts[2:])
        formatted_labels.append(f"{line1},\n{line2}")
    else:
        formatted_labels.append(name)

# Apply new labels, remove rotation, center align
ax2.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=18)
# --- END MODIFICATION ---

ax2.tick_params(axis='y', labelsize=18)
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=20)

plt.tight_layout()
plt.savefig('mutation_propensity_difference_sorted.jpg', dpi=300, bbox_inches='tight')
print("Saved figure: mutation_propensity_difference_sorted.jpg")
plt.close()

# Summary statistics table
print("\n" + "="*80)
print("SUMMARY: TREATMENTS RANKED BY MEDIAN DIFFERENCE")
print("="*80)
print(f"\n{'Rank':<6} {'Treatment':<20} {'Median Δm':<12} {'Mean Δm':<12} {'Std Δm':<12}")
print("-"*65)

for rank, treatment in enumerate(treatments_sorted, 1):
    diff = differences[treatment]
    print(f"{rank:<6} {treatment:<20} {np.median(diff):<12.3f} {np.mean(diff):<12.3f} {np.std(diff):<12.3f}")
print("-"*65)
print("\nDone.")
