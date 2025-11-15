import pickle
import matplotlib.pyplot as plt
import glob
from joblib import Parallel, delayed
import numpy as np
import pickle



dirr = 'results_1000'

#results/simualtion_res_treatment{treatment}_beta{beta}_LOO{LOO}.p

fnames = glob.glob(f'{dirr}/*')

print(len(fnames))



PR_consensus = 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'

PR_consensus = np.array(list(PR_consensus))


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

    fname = f'treatment{treatment}_PR_evol_onehot_logistic.p'
    with open(fname, 'rb') as dbfile:
        info = pickle.load(dbfile)

    PR_onehot_encoder = info['onehot_encoder']

    onehot_inv_map[treatment] = dict()

    bit_idx = 0

    for pos, cat in enumerate(PR_onehot_encoder.categories_):

        for AA in cat:
            onehot_inv_map[treatment][bit_idx] = (pos, AA)

            bit_idx += 1



from tqdm import tqdm
# Figure 3 to show scatter plot of log(-log prog to reach drug res) and mutation propensity

# Proposal: log(-log prog to reach drug res) -> 'Drug Susceptibility'

example_treatment = 'IDV,RTV,SQV'
example_drug_res = 'NFV'


all_unique_fitness = []
all_unique_log_log_prob = []

for fname in tqdm(fnames):

    treatment, beta, LOO = parse_fname(fname)

    if treatment != example_treatment:
        continue

    if LOO is not None:
        continue

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
        walk_drug_res = walk['walk_drug_res'][example_drug_res][0]


        drug_res_events = np.argwhere(walk_drug_res==1).flatten()

        drug_res_genotypes = walk_seqs[drug_res_events]
        drug_res_FEM_fitness = walk_mutation_propensity[drug_res_events]
        drug_res_log_log_prob_to_reach = walk_log_log_prob_to_reach[drug_res_events]

        drug_res_unique_genotypes, drug_res_unique_indices = np.unique(drug_res_genotypes, axis=0, return_index=True)
        drug_res_unique_FEM_fitness = drug_res_FEM_fitness[drug_res_unique_indices]
        drug_res_unique_log_log_prob_to_reach = drug_res_log_log_prob_to_reach[drug_res_unique_indices]


        all_unique_fitness.append(drug_res_unique_FEM_fitness)
        all_unique_log_log_prob.append(drug_res_unique_log_log_prob_to_reach)


all_unique_fitness = np.hstack(all_unique_fitness)
all_unique_log_log_prob = np.hstack(all_unique_log_log_prob)


fname = f'all_unique_stats_treatment{example_treatment}_drug{example_drug_res}.npz'

np.savez(fname, all_unique_fitness=all_unique_fitness, all_unique_log_log_prob=all_unique_log_log_prob)



