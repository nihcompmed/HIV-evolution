import pickle
import numpy as np
import helper_functions_v2 as hf
from tqdm import tqdm
import sys

treatment = sys.argv[1]


# Load fams
dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()

# Load onehot dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

n_cpu = 12

FEM_dict = dict()


#for treatment in tqdm(do_fams):

seqs = do_fams[treatment]

onehot_seqs = []

for seq in seqs:

    onehot_seq = hf.seq_to_onehot(seq, onehot_dict)
    onehot_seqs.append(onehot_seq)

onehot_seqs = np.vstack(onehot_seqs)


FEM_dict = dict()

W, B = hf.predict_full_w(onehot_seqs
            , onehot_dict['boundaries']\
            , n_cpu)

FEM_dict['WW'] = W
FEM_dict['BB'] = B


dbfile = open(f'FEM_prot_evolution_{treatment}.p', 'wb')
pickle.dump(FEM_dict, dbfile)
dbfile.close()
