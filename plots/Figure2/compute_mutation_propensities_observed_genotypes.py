import pickle
import numpy as np
from collections import defaultdict
import glob
import os
from tqdm import tqdm

"""
Compute mutation propensity of OBSERVED genotypes within each treatment regimen

For each treatment F:
  - Take observed sequences from that treatment
  - Compute m_F(σ) = -min_i log P_F(A_i = σ_i | σ_∼i) using models trained on F

This measures how well observed sequences conform to their own treatment's learned patterns

Output: mutation_propensity_observed_results.p
"""

print("\n" + "="*80)
print("MUTATION PROPENSITY ANALYSIS: OBSERVED GENOTYPES")
print("="*80)


fname = '../../preprocess_data/PI_treatments_dict.p'
with open(fname, 'rb') as dbfile:
    PI_treatments = pickle.load(dbfile)

def compute_mutation_propensity(seq, models_dict, encoder, boundaries):
    """
    Compute mutation propensity for a single sequence
    
    m_F(σ) = -min_i log P_F(A_i = σ_i | σ_∼i)
    
    Returns:
        mutation_propensity: scalar value (or np.nan if cannot compute)
    """
    # Encode sequence
    try:
        onehot_seq = encoder.transform([seq]).toarray()[0]
    except:
        return np.nan
    
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


if __name__ == "__main__":
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
    for f_path in tqdm(model_files, desc="Processing Treatments"):
        try:
            fname = os.path.basename(f_path)
            treatment_name = fname.split('_PR_evol_onehot_logistic.p')[0].replace('treatment', '')
        except Exception:
            treatment_name = os.path.basename(f_path)

        try:
            with open(f_path, 'rb') as dbfile:
                info = pickle.load(dbfile)
        except Exception as e:
            print(f"Warning: Could not load or process file {f_path}. Skipping. Error: {e}")
            continue

        encoder = info['onehot_encoder']
        models = info['logistic_regression']
        lenns = np.array([len(cat) for cat in encoder.categories_])
        boundaries = np.hstack([[0], np.cumsum(lenns)])

        observed_seqs = PI_treatments[treatment_name]
        print(f"  Observed sequences: {len(observed_seqs)}")
        
        # Compute mutation propensity for each observed sequence
        propensities = []
        n_failed = 0
        
        print(f"  Computing mutation propensities...")
        for seq_idx, seq in enumerate(observed_seqs):
            if (seq_idx + 1) % 1000 == 0:
                print(f"    Progress: {seq_idx + 1}/{len(observed_seqs)}")
            
            mp = compute_mutation_propensity(seq, models, encoder, boundaries)
            
            if not np.isnan(mp):
                propensities.append(mp)
            else:
                n_failed += 1
        
        propensities = np.array(propensities)
        mutation_propensities_observed[treatment_name] = propensities
        
        print(f"  Successfully computed: {len(propensities)}/{len(observed_seqs)}")
        if n_failed > 0:
            print(f"  Failed to compute: {n_failed}")


        propensities = np.array(propensities)
        mutation_propensities_observed[treatment_name] = propensities

    
    with open('mutation_propensity_observed_results.p', 'wb') as f:
        pickle.dump(mutation_propensities_observed, f)




