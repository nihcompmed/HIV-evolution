from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
from joblib import Parallel, delayed

"""
Train epistatic (logistic regression) and independent (frequency) models
on FULL dataset for each treatment.

These models will be used for:
1. Evolutionary trajectory simulations
2. Trajectory realism validation

Output: One pickle file per treatment containing:
- Trained logistic regression models (epistatic)
- Frequency distributions (independent)
- Encoder and metadata
"""

# Load data
print("Loading data...")
main_dirr = '/Users/aggarwalm4/Downloads/HIV_revision/FINAL_REVISION_FOLDER/'
fname = f'{main_dirr}/data/PI_treatments_dict.p'
with open(fname, 'rb') as dbfile:
    PI_treatments = pickle.load(dbfile)

# Parameters
MIN_SEQUENCES = 100  # Only train for treatments with sufficient data

# Print summary
print("\n" + "="*80)
print("TRAINING MODELS ON FULL DATASET")
print("="*80)
print(f"{'Treatment':<20} {'N Sequences':<15} {'Status':<30}")
print("-"*80)

treatments_to_train = []
for treatment in sorted(PI_treatments.keys()):
    n_seqs = len(PI_treatments[treatment])
    if n_seqs >= MIN_SEQUENCES:
        status = f"✓ Will train"
        treatments_to_train.append(treatment)
    else:
        status = f"✗ Skip (< {MIN_SEQUENCES} sequences)"
    print(f"{treatment:<20} {n_seqs:<15} {status:<30}")

print("-"*80)
print(f"Total treatments: {len(PI_treatments)}")
print(f"Will train: {len(treatments_to_train)}")
print(f"Will skip: {len(PI_treatments) - len(treatments_to_train)}")
print("="*80 + "\n")

# Train models for each treatment
all_trained_models = {}

for treatment in PI_treatments:
    seqs = PI_treatments[treatment]
    n_seqs = len(seqs)
    
    # Skip if too few sequences
    if n_seqs < MIN_SEQUENCES:
        continue
    
    print(f"\n{'='*80}")
    print(f"Training models for treatment: {treatment}")
    print(f"{'='*80}")
    print(f"Total sequences: {n_seqs}")
    
    # Fit encoder on ALL data
    print("  Fitting OneHotEncoder...")
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(seqs)
    
    # Get onehot for all sequences
    onehot_seqs = np.array(encoder.transform(seqs).todense())
    
    # Compute boundaries for each position
    lenns = np.array([len(cat) for cat in encoder.categories_])
    boundaries = np.hstack([[0], np.cumsum(lenns)])
    
    print(f"  Sequence length: {len(seqs[0])} positions")
    print(f"  One-hot dimension: {onehot_seqs.shape[1]}")
    
    # Train EPISTATIC models (one per position)
    def train_position_epistatic(pos):
        """Train logistic regression for position pos"""
        if lenns[pos] == 1:
            # Only one category at this position - skip
            return pos, None, None
        
        targets = seqs[:, pos]
        unique_classes = np.unique(targets)
        
        if len(unique_classes) < 2:
            # Only one class - can't train classifier
            return pos, None, None
        
        # Features: all positions EXCEPT pos
        start = boundaries[pos]
        end = boundaries[pos + 1]
        X = np.hstack([onehot_seqs[:, :start], onehot_seqs[:, end:]])
        
        # Train model
        model = LogisticRegression(penalty='l2', max_iter=10000, random_state=42)
        model.fit(X, targets)
        
        return pos, model, unique_classes
    
    print("  Training epistatic models (logistic regression)...")
    epistatic_results = Parallel(n_jobs=12, verbose=10)(
        delayed(train_position_epistatic)(pos) for pos in range(len(seqs[0]))
    )
    
    # Collect trained models
    epistatic_models = {}
    epistatic_classes = {}
    
    for pos, model, classes in epistatic_results:
        if model is not None:
            epistatic_models[pos] = model
            epistatic_classes[pos] = classes
    
    print(f"  Trained {len(epistatic_models)} epistatic models")
    
    # Compute INDEPENDENT model (frequency distributions)
    print("  Computing independent model (frequencies)...")
    independent_freqs = {}
    
    for pos in range(len(seqs[0])):
        if lenns[pos] == 1:
            continue
        
        targets = seqs[:, pos]
        unique, counts = np.unique(targets, return_counts=True)
        
        if len(unique) < 2:
            continue
        
        freqs = dict(zip(unique, counts / len(targets)))
        independent_freqs[pos] = freqs
    
    print(f"  Computed {len(independent_freqs)} frequency distributions")
    
    # Store all information for this treatment
    all_trained_models[treatment] = {
        'n_seqs': n_seqs,
        'seq_length': len(seqs[0]),
        'encoder': encoder,
        'boundaries': boundaries,
        'category_lengths': lenns,
        'epistatic_models': epistatic_models,
        'epistatic_classes': epistatic_classes,
        'independent_freqs': independent_freqs,
        'positions_with_models': list(epistatic_models.keys())
    }
    
    # Save individual treatment file
    treatment_filename = f'trained_models_{treatment}.p'
    with open(treatment_filename, 'wb') as f:
        pickle.dump(all_trained_models[treatment], f)
    
    print(f"  Saved to: {treatment_filename}")
    
    # Summary
    print(f"\n  Summary for {treatment}:")
    print(f"    Sequences used: {n_seqs}")
    print(f"    Positions: {len(seqs[0])}")
    print(f"    Epistatic models trained: {len(epistatic_models)}")
    print(f"    Independent distributions: {len(independent_freqs)}")
    print(f"    Positions trainable: {len(epistatic_models)}/{len(seqs[0])}")

# Save combined file with all treatments
print("\n" + "="*80)
print("Saving combined file with all treatments...")
with open('all_trained_models.p', 'wb') as f:
    pickle.dump(all_trained_models, f)

print("Saved to: all_trained_models.p")
print("="*80)

# Print final summary
print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\nTrained models summary:")
print("-"*80)
print(f"{'Treatment':<20} {'N Seqs':<10} {'Epistatic':<15} {'Independent':<15}")
print("-"*80)

for treatment in sorted(all_trained_models.keys()):
    info = all_trained_models[treatment]
    n_seqs = info['n_seqs']
    n_epistatic = len(info['epistatic_models'])
    n_independent = len(info['independent_freqs'])
    print(f"{treatment:<20} {n_seqs:<10} {n_epistatic:<15} {n_independent:<15}")

print("-"*80)
print("\nFiles created:")
print("  - all_trained_models.p (combined)")
for treatment in sorted(all_trained_models.keys()):
    print(f"  - trained_models_{treatment}.p")
print("="*80)



