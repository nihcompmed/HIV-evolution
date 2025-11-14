import pickle
import numpy as np
import helper_functions_v2 as hf
from tqdm import tqdm


dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()

dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

all_onehot_seqs = []

for treatment in do_fams:

    # Load seqs
    seqs = do_fams[treatment]

    onehot_seqs = []

    for seq in seqs:

        onehot_seq = hf.seq_to_onehot(seq, onehot_dict)
        onehot_seqs.append(onehot_seq)

    onehot_seqs = np.vstack(onehot_seqs)

    all_onehot_seqs.append(onehot_seqs)

all_onehot_seqs = np.vstack(all_onehot_seqs)

log_probs_dict = dict()


for treatment in tqdm(do_fams):

    # Load background FEM

    dbfile = open(f'FEM_prot_evolution_{treatment}.p', 'rb')
    FEM_dict = pickle.load(dbfile)
    WW = FEM_dict['WW']
    BB = FEM_dict['BB']
    dbfile.close()


    # Get prob vector

    log_probs = hf.get_log_probs_multiple_seqs(all_onehot_seqs, WW, BB, onehot_dict['boundaries'])

    log_probs_dict[treatment] = log_probs



dbfile = open(f'log_probs_dict.p', 'wb')
pickle.dump(log_probs_dict, dbfile)
dbfile.close()





