import pickle
import numpy as np
from collections import defaultdict
import glob
import os
from tqdm import tqdm

"""
Compute mutation propensity of drug-naive sequences across ALL treatment regimens
INCLUDING 'None' treatment

Mutation propensity: m_F(σ) = -min_i log P_F(A_i = σ_i | σ_∼i)
Lower values = more consistent with treatment-specific patterns

Output: mutation_propensity_results.p
"""

print("\n" + "="*80)
print("MUTATION PROPENSITY ANALYSIS: DRUG-NAIVE SEQUENCES")
print("="*80)



# Configuration
model_dir = "../../train_models/models/" # ASSUMES your .p files are in a 'models' subfolder

mutation_propensities_observed = {}

# --- 2. Find Model Files ---
model_files = glob.glob(os.path.join(model_dir, "*_PR_evol_onehot_logistic.p"))
if not model_files:
    print(f"Error: No model files found in directory: {model_dir}")
    print("Please ensure your .p files are in that folder.")
    exit()

# --- 4. Process Each Treatment ---
all_trained_models = dict()
for f_path in tqdm(model_files, desc="Processing Treatments"):
    try:
        fname = os.path.basename(f_path)
        treatment_name = fname.split('_PR_evol_onehot_logistic.p')[0].replace('treatment', '')
    except Exception:
        treatment_name = os.path.basename(f_path)

    try:
        with open(f_path, 'rb') as dbfile:
            info = pickle.load(dbfile)
            all_trained_models[treatment_name] = dict()
            all_trained_models[treatment_name]['encoder'] = info['onehot_encoder']
            all_trained_models[treatment_name]['epistatic_models'] = info['logistic_regression']

            lenns = np.array([len(cat) for cat in info['onehot_encoder'].categories_])
            boundaries = np.hstack([[0], np.cumsum(lenns)])

            all_trained_models[treatment_name]['boundaries'] = boundaries

    except Exception as e:
        print(f"Warning: Could not load or process file {f_path}. Skipping. Error: {e}")
        continue

# Get list of ALL treatments with trained models (INCLUDING 'None')
treatments_with_models = sorted(all_trained_models.keys())
print(f"Treatment regimens to analyze: {len(treatments_with_models)}")
for t in treatments_with_models:
    print(f"  - {t}")


# Load data
print("\nLoading data...")

fname = '../../preprocess_data/PI_treatments_dict.p'
with open(fname, 'rb') as dbfile:
    PI_treatments = pickle.load(dbfile)


drug_naive_seqs = PI_treatments['None']
print(f"\nDrug-naive sequences: {len(drug_naive_seqs)}")


def can_encode_sequence(seq, encoder):
    """
    Check if sequence can be encoded by the given encoder
    Returns True if all amino acids in seq are in encoder's vocabulary
    """
    for pos, aa in enumerate(seq):
        if pos >= len(encoder.categories_):
            return False
        if aa not in encoder.categories_[pos]:
            return False
    return True

# Test each drug-naive sequence
encodable_in_all = []
encodable_counts = defaultdict(int)

print(f"\nTesting {len(drug_naive_seqs)} drug-naive sequences...")

for seq_idx, seq in enumerate(drug_naive_seqs):
    if (seq_idx + 1) % 1000 == 0:
        print(f"  Progress: {seq_idx + 1}/{len(drug_naive_seqs)}")
    
    # Check if this sequence can be encoded in all treatments
    can_encode_everywhere = True
    
    for treatment in treatments_with_models:
        encoder = all_trained_models[treatment]['encoder']
        
        if not can_encode_sequence(seq, encoder):
            can_encode_everywhere = False
            break
        
        encodable_counts[treatment] += 1
    
    if can_encode_everywhere:
        encodable_in_all.append(seq)

print(f"\nResults:")
print(f"  Sequences encodable in ALL treatments: {len(encodable_in_all)}")
print(f"\nEncodable counts by treatment:")
for treatment in sorted(encodable_counts.keys()):
    print(f"  {treatment:<20}: {encodable_counts[treatment]}/{len(drug_naive_seqs)} "
          f"({100*encodable_counts[treatment]/len(drug_naive_seqs):.1f}%)")

if len(encodable_in_all) == 0:
    print("\nERROR: No sequences encodable in all treatments!")
    print("Cannot proceed with analysis.")
    exit(1)

print(f"\n{len(encodable_in_all)} sequences will be used for mutation propensity analysis")

# Convert to numpy array
encodable_seqs = np.array(encodable_in_all)

# Step 2: Compute mutation propensity for each sequence in each treatment
print("\n" + "="*80)
print("STEP 2: Computing mutation propensities")
print("="*80)

def compute_mutation_propensity(seq, models_dict, encoder, boundaries):
    """
    Compute mutation propensity for a single sequence in a treatment
    
    m_F(σ) = -min_i log P_F(A_i = σ_i | σ_∼i)
    
    Returns:
        mutation_propensity: scalar value
    """
    # Encode sequence
    onehot_seq = encoder.transform([seq]).toarray()[0]
    
    # Compute log probability at each position
    log_probs = []
    
    for pos in models_dict.keys():
        model = models_dict[pos]
        
        # Get features (all positions except pos)
        start_boundary = boundaries[pos]
        end_boundary = boundaries[pos + 1]
        X = np.hstack([onehot_seq[:start_boundary], onehot_seq[end_boundary:]])
        
        # Get log probability of observed amino acid
        log_probs_pos = model.predict_log_proba([X])[0]
        
        # Find index of observed amino acid
        observed_aa = seq[pos]
        if observed_aa in model.classes_:
            aa_idx = np.where(model.classes_ == observed_aa)[0][0]
            log_prob = log_probs_pos[aa_idx]
            log_probs.append(log_prob)
    
    if len(log_probs) == 0:
        return np.nan
    
    # Mutation propensity = -min(log_probs)
    mutation_propensity = -np.min(log_probs)
    
    return mutation_propensity

# Compute mutation propensities for ALL treatments (including 'None')
mutation_propensities = {treatment: [] for treatment in treatments_with_models}

print(f"\nComputing for {len(encodable_seqs)} sequences across {len(treatments_with_models)} treatments...")

for treatment in treatments_with_models:
    print(f"\n  Treatment: {treatment}")
    
    models_dict = all_trained_models[treatment]['epistatic_models']
    encoder = all_trained_models[treatment]['encoder']
    boundaries = all_trained_models[treatment]['boundaries']
    
    propensities = []
    
    for seq_idx, seq in enumerate(encodable_seqs):
        if (seq_idx + 1) % 5000 == 0:
            print(f"    Progress: {seq_idx + 1}/{len(encodable_seqs)}")
        
        mp = compute_mutation_propensity(seq, models_dict, encoder, boundaries)
        
        if not np.isnan(mp):
            propensities.append(mp)
    
    mutation_propensities[treatment] = np.array(propensities)
    
    print(f"    Computed {len(propensities)} propensities")
    print(f"    Mean: {np.mean(propensities):.3f}, Median: {np.median(propensities):.3f}, "
          f"Std: {np.std(propensities):.3f}")

# Statistical summary
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)
print(f"\n{'Treatment':<20} {'N':<10} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-"*90)

for treatment in treatments_with_models:
    vals = mutation_propensities[treatment]
    print(f"{treatment:<20} {len(vals):<10} {np.mean(vals):<10.3f} {np.median(vals):<10.3f} "
          f"{np.std(vals):<10.3f} {np.min(vals):<10.3f} {np.max(vals):<10.3f}")

print("-"*90)

# Save results
print("\n" + "="*80)
print("Saving results...")
results = {
    'encodable_sequences': encodable_seqs,
    'n_encodable': len(encodable_seqs),
    'n_drug_naive_total': len(drug_naive_seqs),
    'treatments': treatments_with_models,
    'mutation_propensities': mutation_propensities,
    'summary_stats': {
        treatment: {
            'n': len(mutation_propensities[treatment]),
            'mean': float(np.mean(mutation_propensities[treatment])),
            'median': float(np.median(mutation_propensities[treatment])),
            'std': float(np.std(mutation_propensities[treatment])),
            'min': float(np.min(mutation_propensities[treatment])),
            'max': float(np.max(mutation_propensities[treatment]))
        }
        for treatment in treatments_with_models
    }
}

fname = 'mutation_propensity_drug_naive_results.p'

with open(fname, 'wb') as f:
    pickle.dump(results, f)

print(f"Saved to: {fname}")
print("="*80)

print("\n" + "="*80)
print("COMPUTATION COMPLETE")
print("="*80)
print(f"\nAnalyzed {len(encodable_seqs)} drug-naive sequences")
print(f"across {len(treatments_with_models)} treatment regimens (including 'None')")
print("="*80)



