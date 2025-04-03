import pickle
import matplotlib.pyplot as plt
import helper_functions_v2 as hf
import numpy as np
import scipy
from joblib import Parallel, delayed
from tqdm import tqdm
import sys


background_treatment = sys.argv[1]
beta = float(sys.argv[2])

# Load onehot dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

VIDX_map = onehot_dict['VIDX_map']
n_bits = onehot_dict['n_bits']


# Load fams
dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()

drug_FEM_dict = dict()

for drug in ['NFV', 'IDV', 'ATV']:

    fname = f'FEM_drugres_{drug}.npz'
    drug_res_FEM = np.load(fname)
    WW = drug_res_FEM['weights']
    BB = drug_res_FEM['bias']

    drug_FEM_dict[drug] = dict()
    drug_FEM_dict[drug]['WW'] = WW
    drug_FEM_dict[drug]['BB'] = BB


all_info_dict = dict()


all_info_dict = dict()
for drug in drug_FEM_dict:
    all_info_dict[drug] = dict()

print(f'Doing {background_treatment} beta {beta}...')

traj_fname = f'direct_results/trajs1000_{background_treatment}_beta{beta}.p'

dbfile = open(traj_fname, 'rb')
trajs = pickle.load(dbfile)
dbfile.close()


for single_traj in tqdm(trajs):

    first_seq_onehot, mutations, scores = single_traj

    traj_seqs_onehot = [first_seq_onehot]

    for mutate_idx in mutations:

        last_seq_onehot = traj_seqs_onehot[-1]
        last_seq_sparse = hf.onehot_to_sparse(last_seq_onehot)

        # Mutate
        mutate_vidx = VIDX_map[mutate_idx]
        last_seq_sparse[mutate_vidx] = mutate_idx

        traj_seqs_onehot.append(hf.sparse_to_onehot(last_seq_sparse, n_bits))


    traj_seqs_onehot = np.vstack(traj_seqs_onehot)

    scores = np.array(scores)
    cum_log_probs = np.cumsum(scores)

    for drug in drug_FEM_dict:
    
        WW = drug_FEM_dict[drug]['WW']
        BB = drug_FEM_dict[drug]['BB']

        HH = np.matmul(traj_seqs_onehot, WW) + BB

        prob_drug_res_true = scipy.special.expit(HH)

        idxs = np.argwhere(prob_drug_res_true >= 0.5).flatten()

        for idx in idxs:
            arr_seq = traj_seqs_onehot[idx]
            string_seq = ''.join([str(x) for x in arr_seq])

            # Assuming that first seq is not drug res, which is true here because it is consensus
            this_cumulative_log_probs = cum_log_probs[idx-1]

            if string_seq not in all_info_dict[drug]:
                all_info_dict[drug][string_seq] = this_cumulative_log_probs
            else:
                all_info_dict[drug][string_seq] =\
                        max(all_info_dict[drug][string_seq]\
                                , this_cumulative_log_probs)
            

fname = f'mutations_treat{background_treatment}_prob_to_reach_drug_res_beta{beta}.p'
print('Saving...')
dbfile = open(fname, 'wb')
pickle.dump(all_info_dict, dbfile)
dbfile.close()
    

            









