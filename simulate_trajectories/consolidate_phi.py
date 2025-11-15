
from joblib import Parallel, delayed
import math
import pickle
import matplotlib.pyplot as plt
import glob
import numpy as np

model_dirr = '../train_models/models'

results_dirr = 'results_1000'

fnames = glob.glob(f'{results_dirr}/*')

print(len(fnames))


PR_consensus = 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'

PR_consensus = np.array(list(PR_consensus))


normalization_factor_prob_to_reach = 10
normalization_mutation_propensity = 20

def parse_fname(fname):

    stem = fname.split('/')[-1]

    parser = stem.split('_')

    treatment = parser[2][9:]
    beta = float(parser[3][4:])

    LOO = parser[-1].split('.')[0]

    if LOO[0] == 'L':
        LOO = int(LOO[3:])
    else:
        LOO = None

    
    
    return treatment, beta, LOO
    


# Get onehot_inv_maps of all treatments to generate mutation genotypes

fname = 'PI_treatments_dict.p'
with open(fname, 'rb') as dbfile:
    PI_treatments = pickle.load(dbfile)


onehot_inv_map = dict()

for treatment in PI_treatments:

    fname = f'{model_dirr}/treatment{treatment}_PR_evol_onehot_logistic.p'
    with open(fname, 'rb') as dbfile:
        info = pickle.load(dbfile)

    PR_onehot_encoder = info['onehot_encoder']

    onehot_inv_map[treatment] = dict()

    bit_idx = 0

    for pos, cat in enumerate(PR_onehot_encoder.categories_):

        for AA in cat:
            onehot_inv_map[treatment][bit_idx] = (pos, AA)

            bit_idx += 1




def single_fname(fname):
  
    treatment, beta, LOO = parse_fname(fname)

    if LOO is None:
        LOO = -1

    experiment_unique_fitness = dict()
    experiment_unique_log_log_prob = dict()


    dbfile = open(fname, 'rb')
    data = pickle.load(dbfile)
    dbfile.close()
  
    for walk in data:
  
        walk_mutations = walk['walk_mutations']
        walk_mutation_propensity = np.array(walk['walk_FEM_fitness'])
        walk_log_prob_step = np.array(walk['walk_log_prob_step'])

        walk_log_prob_to_reach = np.cumsum(walk_log_prob_step)
        walk_log_log_prob_to_reach = np.log(-walk_log_prob_to_reach)
  
        walk_seqs = []
  
        first_seq = PR_consensus.copy()
  
        this_seq = first_seq.copy()
  
        for mutate_bit_idx in walk_mutations:
          
            pos, AA = onehot_inv_map[treatment][mutate_bit_idx]
  
            this_seq[pos] = AA
  
            walk_seqs.append(this_seq.copy())
  
        walk_seqs = np.vstack(walk_seqs)

        # DRUG resistance

        for drug in walk['walk_drug_res']:

            if drug not in experiment_unique_fitness:
                experiment_unique_fitness[drug] = []
                experiment_unique_log_log_prob[drug] = []
              
            walk_drug_res = walk['walk_drug_res'][drug][0]
  
  
            drug_res_events = np.argwhere(walk_drug_res==1).flatten()
  
            drug_res_genotypes = walk_seqs[drug_res_events]
            drug_res_FEM_fitness = walk_mutation_propensity[drug_res_events]
            drug_res_log_log_prob_to_reach = walk_log_log_prob_to_reach[drug_res_events]
  
            drug_res_unique_genotypes, drug_res_unique_indices = np.unique(drug_res_genotypes, axis=0, return_index=True)
          
            drug_res_unique_FEM_fitness = drug_res_FEM_fitness[drug_res_unique_indices]
            drug_res_unique_log_log_prob_to_reach = drug_res_log_log_prob_to_reach[drug_res_unique_indices]

            experiment_unique_fitness[drug].append(drug_res_unique_FEM_fitness)
            experiment_unique_log_log_prob[drug].append(drug_res_unique_log_log_prob_to_reach)


    experiment_phi = dict()
  
    for drug in experiment_unique_fitness:

        this_experiment_unique_fitness = np.hstack(experiment_unique_fitness[drug])
        this_experiment_unique_log_log_prob = np.hstack(experiment_unique_log_log_prob[drug])
      
        scaled_fitness = this_experiment_unique_fitness/normalization_mutation_propensity
        scaled_prob_to_reach = this_experiment_unique_log_log_prob/normalization_factor_prob_to_reach

        phi_norm = np.sqrt(scaled_prob_to_reach**2 + scaled_fitness**2)

        hist, _ = np.histogram(phi_norm, bins=1000, range=(0, math.sqrt(2)))

        experiment_phi[drug] = hist


    return treatment, beta, LOO, experiment_phi


res = Parallel(n_jobs=24, verbose=4)\
        (delayed(single_fname)(fname) for fname in fnames)

# IMPORTANT TO COMMENT IF ALREADY COMPUTED!

fname = 'consolidated_binned_phi_1000.p'
dbfile = open(fname, 'wb')
pickle.dump(res, dbfile)
dbfile.close()
