import pickle
import matplotlib.pyplot as plt
import helper_functions_v2 as hf
import numpy as np
import scipy
from tqdm import tqdm


# Load onehot dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

boundaries = onehot_dict['boundaries']

# Load fams
dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()

all_info_info_dict = dict()

for background_treatment in do_fams:

    fname = f'FEM_prot_evolution_{background_treatment}.p'
    
    dbfile = open(fname, 'rb')
    FEM_dict = pickle.load(dbfile)
    dbfile.close()

    WW = FEM_dict['WW']
    BB = FEM_dict['BB']

    all_info_info_dict[background_treatment] = dict()

    #for beta in [0.2, 0.4, 0.6, 0.8, 1.0]:
    for beta in [0.4, 0.6, 0.8, 1.0]:
    #for beta in [0.8, 1.0]:
    #for beta in [1.0]:


        fname = f'mutations_treat{background_treatment}_prob_to_reach_drug_res_beta{beta}.p'
        dbfile = open(fname, 'rb')
        all_info_dict = pickle.load(dbfile)
        dbfile.close()


        for drug in all_info_dict:

            print(f'Doing {background_treatment} beta {beta} drug {drug}')

            if drug not in all_info_info_dict[background_treatment]:

                all_info_info_dict[background_treatment][drug] = dict()

            these_seqs = all_info_dict[drug]

            for string_seq in tqdm(these_seqs):

                if string_seq in all_info_info_dict[background_treatment][drug]:
                    all_info_info_dict[background_treatment][drug][string_seq][0] =\
                            max(all_info_info_dict[background_treatment][drug][string_seq][0]\
                                    , all_info_dict[drug][string_seq])
                else:

                    onehot_seq = np.array(list(string_seq), dtype=int)

                    log_probs = hf.get_log_probs_single_seq(onehot_seq, WW, BB, boundaries)

                    sparse_seq = hf.onehot_to_sparse(onehot_seq)

                    current_log_probs = log_probs[sparse_seq]

                    all_info_info_dict[background_treatment][drug][string_seq] =\
                                            [all_info_dict[drug][string_seq]\
                                                , np.amin(current_log_probs)]





fname = 'direct_mutations_prob_to_reach_drug_res_and_stay.p'
print('Saving...')
dbfile = open(fname, 'wb')
pickle.dump(all_info_info_dict, dbfile)
dbfile.close()
    



