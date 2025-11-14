from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from joblib import Parallel, delayed
import sys


#treatment = 'None'
#beta = 0.40
#LOO = n_bits

treatment = sys.argv[1]
beta = float(sys.argv[2])
LOO = int(sys.argv[3])


PR_consensus = 'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF'

PR_consensus = np.array(list(PR_consensus))


n_walks = 1000
n_steps = 1000

# Load drug logistic regression models

# Drug resistance
fname = 'PI_drug_res_dict.p'
dbfile = open(fname, 'rb')
PR_drug_res = pickle.load(dbfile)
dbfile.close()

drug_logistic_regression = dict()

for drug in PR_drug_res:

    fname = f'treatment{treatment}_drugres{drug}_onehot_logistic.p'
    dbfile = open(fname, 'rb')
    drug_logistic_regression[drug] = pickle.load(dbfile)
    dbfile.close()

fname = f'treatment{treatment}_PR_evol_onehot_logistic.p'
dbfile = open(fname, 'rb')
info = pickle.load(dbfile)
dbfile.close()

PR_onehot_encoder = info['onehot_encoder']
PR_logistic_regression = info['logistic_regression']

lenns = []

onehot_map = dict()
onehot_inv_map = dict()

pos_info = dict()

bit_idx = 0 

non_varying_bits = []

for pos, cat in enumerate(PR_onehot_encoder.categories_):

    lenns.append(len(cat))

    for AA in cat:
        onehot_map[(pos, AA)] = bit_idx
        onehot_inv_map[bit_idx] = (pos, AA)

        if lenns[-1] == 1:
            non_varying_bits.append(bit_idx)
            
        bit_idx += 1

non_varying_bits = np.array(non_varying_bits, dtype=int)

n_bits = bit_idx

print('total bits', n_bits)
print('non varying bits', non_varying_bits)

lenns = np.array(lenns)

boundaries = np.hstack([[0], np.cumsum(lenns)])

#for walk in range(n_walks):
def single_walk(walk\
                , PR_onehot_encoder\
                , boundaries\
                , onehot_map\
                , onehot_inv_map\
                , non_varying_bits\
                , PR_logistic_regression\
                , drug_logistic_regression):

    first_seq = PR_consensus.copy()
    
    this_seq = first_seq.copy()
    
    # Get FEM_fitness of this sequence
    def get_seq_logprobs(seq, onehot_encoder, boundaries, logistic_regression):
    
        seq_log_probs = np.zeros((len(seq),))
    
        onehot_seq = np.array(onehot_encoder.transform(seq.reshape(1,-1)).todense())[0]
    
        for pos in range(len(boundaries)-1):
    
            if lenns[pos] == 1:
                continue
    
            this_pos_model = logistic_regression[pos]
    
            this_pos_classes = list(this_pos_model.classes_)
    
            start = boundaries[pos]
            end = boundaries[pos+1]
    
            xx = np.hstack([onehot_seq[:start], onehot_seq[end:]])
    
            pos_log_probs = this_pos_model.predict_log_proba(xx.reshape(1,-1))[0]
    
            this_AA = seq[pos]
    
            this_AA_log_prob = pos_log_probs[this_pos_classes.index(this_AA)]
    
            seq_log_probs[pos] = this_AA_log_prob
    
        return seq_log_probs
    
    def get_FEM_fitness(seq, onehot_encoder, boundaries, logistic_regression):
    
        seq_log_probs = get_seq_logprobs(seq, onehot_encoder, boundaries, logistic_regression)
    
        return abs(np.min(seq_log_probs))
    
    walk_seqs = []
    walk_mutations = []
    walk_FEM_fitness = []
    walk_log_prob_step = []
    walk_drug_res = dict()
    
    for drug in PR_drug_res:
    
        walk_drug_res[drug] = []
    
    for ww in range(n_steps):
    
        onehot_seq = np.array(PR_onehot_encoder.transform(this_seq.reshape(1,-1)).todense())[0]
    
        all_log_probs = np.zeros((n_bits,))

        seq_log_probs = np.zeros((len(this_seq),))
    
        for pos in range(len(boundaries)-1):
    
            if lenns[pos] == 1:
                continue
    
            this_pos_model = PR_logistic_regression[pos]
    
            start = boundaries[pos]
            end = boundaries[pos+1]
    
            xx = np.hstack([onehot_seq[:start], onehot_seq[end:]])
    
            pos_log_probs = this_pos_model.predict_log_proba(xx.reshape(1,-1))[0]

            this_pos_classes = list(this_pos_model.classes_)
            
            this_AA = this_seq[pos]

            for AA, log_prob in zip(this_pos_model.classes_, pos_log_probs):
    
                bit_idx = onehot_map[(pos, AA)]
    
                all_log_probs[bit_idx] = log_prob

                if AA == this_AA:
                    seq_log_probs[pos] = log_prob

        walk_FEM_fitness.append(abs(np.min(seq_log_probs)))
    
        all_log_probs_tempered = beta*all_log_probs
    
        probs = np.exp(all_log_probs_tempered)
    
        # NOTE: probs of non_varying_bits '0'
        if len(non_varying_bits):
            probs[non_varying_bits] = 0
    
        # Do not pick LOO if specified
        if LOO < n_bits:
            probs[LOO] = 0
    
        # Normalize for random.choice
        probs = probs/np.sum(probs)
    
        # Choose
        mutate_bit_idx = np.random.choice(len(probs), p=probs)
    
        pos, AA = onehot_inv_map[mutate_bit_idx]
    
        this_seq[pos] = AA

        walk_mutations.append(mutate_bit_idx)
    
        walk_seqs.append(this_seq.copy())
    
        walk_log_prob_step.append(all_log_probs[mutate_bit_idx])

    walk_seqs = np.vstack(walk_seqs)

    walk_FEM_fitness.append(get_FEM_fitness(this_seq, PR_onehot_encoder, boundaries, PR_logistic_regression))

    walk_FEM_fitness = walk_FEM_fitness[1:]
        
    for drug in PR_drug_res:
    
        seq_onehot_drug = np.array(drug_logistic_regression[drug]['onehot_encoder'].transform(walk_seqs).todense())
    
        seq_drug_res = drug_logistic_regression[drug]['logistic_regression'].predict(seq_onehot_drug)
    
        walk_drug_res[drug].append(seq_drug_res)


        
    res = dict()
    
    res['walk_mutations'] = walk_mutations
    res['walk_FEM_fitness'] = walk_FEM_fitness
    res['walk_log_prob_step'] = walk_log_prob_step
    res['walk_drug_res'] = walk_drug_res

    return res



all_res = Parallel(n_jobs=4, verbose=2)\
            (delayed(single_walk)(walk\
                                , PR_onehot_encoder\
                                , boundaries\
                                , onehot_map\
                                , onehot_inv_map\
                                , non_varying_bits\
                                , PR_logistic_regression\
                                , drug_logistic_regression)
                                for walk in range(n_walks))


if LOO < n_bits:
    fname = f'results_1000/simualtion_res_treatment{treatment}_beta{beta}_LOO{LOO}.p'
else:
    fname = f'results_1000/simualtion_res_treatment{treatment}_beta{beta}_LOO_None.p'

dbfile = open(fname, 'wb')
pickle.dump(all_res, dbfile)
dbfile.close()
    
    
    
    
    
    
    
    
