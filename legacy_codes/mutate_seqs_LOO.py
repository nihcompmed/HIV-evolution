import helper_functions_v2 as hf
import numpy as np
import pickle
from tqdm import tqdm
import sys



treatment = sys.argv[1]
beta = float(sys.argv[2])
LOO = int(sys.argv[3])

###############################
# Load onehot dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

boundaries = onehot_dict['boundaries']
VIDX_map = onehot_dict['VIDX_map']
n_bits = onehot_dict['n_bits']

###############################


###############################
# Mutate consensus

# From https://hivdb.stanford.edu/pages/documentPage/consensus_amino_acid_sequences.html
PR_consensus = np.array([*'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'])
PR_consensus_onehot_seq = hf.seq_to_onehot(PR_consensus, onehot_dict)

###############################

print(f'Doing {treatment}...')

fname = f'FEM_prot_evolution_{treatment}.p'

dbfile = open(fname, 'rb')
FEM_dict = pickle.load(dbfile)
dbfile.close()

WW = FEM_dict['WW']
BB = FEM_dict['BB']

###############################

n_traj = 1000

n_steps = 1000

all_info = []

for traj in tqdm(range(n_traj)):

    first_seq, mutations, scores =\
            hf.mutate_single_tempered(PR_consensus_onehot_seq\
                                    , beta\
                                    , n_steps\
                                    , WW\
                                    , BB\
                                    , boundaries\
                                    , VIDX_map\
                                    , n_bits\
                                    , LOO=LOO)

    all_info.append((first_seq, mutations, scores))


print('Saving dict...')
dbfile = open(f'LOO_results/trajs{n_traj}_{treatment}_LOO{LOO}_beta{beta}.p', 'wb')
pickle.dump(all_info, dbfile)
dbfile.close()


