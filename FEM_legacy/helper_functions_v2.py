import numpy as np
from Bio import SeqIO, BiopythonWarning
import FEM
from joblib import Parallel, delayed
import scipy
import math

def get_seqs_from_fasta(fname):

    seqs = []

    with open(fname, 'r') as f:

        seq_iter = SeqIO.parse(f,'fasta')

        for seq in seq_iter:

            seqs.append(seq)


    return np.array(seqs)


def get_onehot_dict(mat):

    n_cols = mat.shape[1]

    varying_pos = []
    fixed_pos = []

    unique_dict = dict()

    n_bits = 0

    bit_idx = 0
    v_idx = 0

    encode_dict = dict()
    AA_map = []
    VIDX_map = []
    boundaries = [0]

    for pos in range(n_cols):

        this_col = mat[:, pos]

        uni  = np.unique(this_col)

        pair = (pos, uni)

        if len(uni) != 1:

            #print(pos, uni)

            varying_pos.append(pos)
            n_bits += len(uni)

            boundaries.append(boundaries[-1]+len(uni))

            encode_dict[v_idx] = dict()

            for AA in uni:

                AA_map.append(AA)
                VIDX_map.append(v_idx)
                
                encode_dict[v_idx][AA] = bit_idx
                bit_idx += 1

            v_idx += 1

        else:
            fixed_pos.append(pair)

    AA_map = np.array(AA_map)
    VIDX_map = np.array(VIDX_map)
    boundaries = np.array(boundaries)

    onehot_dict = dict()


    onehot_dict['varying_pos'] = varying_pos
    onehot_dict['AA_map'] = AA_map
    onehot_dict['VIDX_map'] = VIDX_map
    onehot_dict['encode_dict'] = encode_dict
    onehot_dict['fixed_pos'] = fixed_pos
    onehot_dict['boundaries'] = boundaries
    onehot_dict['n_bits'] = n_bits



    return onehot_dict



def seq_to_sparse(seq, varying_pos, encode_dict):

    seq_star = seq[varying_pos]

    r_star = np.zeros(len(varying_pos), dtype=int)

    for i, aa in enumerate(seq_star):

        r_star[i] = encode_dict[i][aa]

    return r_star


def sparse_to_onehot(sparse_seq, n_bits):

    onehot_seq = np.zeros(n_bits, dtype=int)
    onehot_seq[sparse_seq] = 1

    return onehot_seq

def onehot_to_sparse(onehot_seq):

    sparse_seq = np.argwhere(onehot_seq==1).flatten()

    return sparse_seq


def seq_to_onehot(seq, onehot_dict):

    varying_pos = onehot_dict['varying_pos']
    encode_dict = onehot_dict['encode_dict']
    n_bits = onehot_dict['n_bits']

    sparse_seq = seq_to_sparse(seq, varying_pos, encode_dict)
    onehot_seq = sparse_to_onehot(sparse_seq, n_bits)

    return onehot_seq

#################################################################


def predict_full_w(onehot_seqs
                , boundaries\
                , n_cpu):


    def single_pos(idx, boundaries, onehot_seqs):

        start = boundaries[idx]
        end = boundaries[idx+1]

        this_xx = np.hstack([onehot_seqs[:, :start], onehot_seqs[:, end:]])

        this_yy = onehot_seqs[:, start:end]

        this_bb, this_ww = FEM.FEM_fit(this_xx, this_yy)

        return start, end, this_ww, this_bb


    N_all = onehot_seqs.shape[1]

    w_fam = np.zeros((N_all, N_all))                                                                     
    b_fam = np.zeros(N_all) 

    res = Parallel(n_jobs=n_cpu, verbose=12)\
            (delayed(single_pos)(idx, boundaries, onehot_seqs) for idx  in range(len(boundaries)-1))

    for rr in res:

        start, end, this_ww, this_bb = rr

        b_fam[start:end] = this_bb                                                                                  
        w_fam[:start,start:end] = this_ww[:start,:]                                                                      
        w_fam[end:,start:end] = this_ww[start:,:]                                                                      


    return w_fam, b_fam



#################################################################

def get_log_probs_multiple_seqs(onehot_seqs, WW, BB, boundaries):

    if onehot_seqs.ndim == 1:
        onehot_seqs = np.expand_dims(onehot_seqs, axis=0)
        
    HH = np.matmul(onehot_seqs, WW) + np.expand_dims(BB, axis=0)

    log_sigmoid = scipy.special.log_expit(-HH)

    log_probs = np.zeros_like(HH)

    for idx in range(len(boundaries)-1):

        start = boundaries[idx]
        end = boundaries[idx+1]

        this_norm = np.sum(log_sigmoid[:, start:end], axis=1)
        
        this_H = HH[:, start:end]

        log_probs[:, start:end] = this_H + np.expand_dims(this_norm, axis=1)

    return log_probs

def get_log_probs_single_seq(onehot_seq, WW, BB, boundaries):

    HH = np.matmul(onehot_seq, WW) + BB

    log_sigmoid = scipy.special.log_expit(-HH)

    log_probs = np.zeros_like(HH)

    for idx in range(len(boundaries)-1):

        start = boundaries[idx]
        end = boundaries[idx+1]

        this_norm = np.sum(log_sigmoid[start:end])
        
        this_H = HH[start:end]

        log_probs[start:end] = this_H + this_norm

    return log_probs

#################################################################


#def mutate_single_direct(onehot_seq, n_steps, WW, BB, boundaries, VIDX_map, n_bits, LOO=None):
#
#    first_seq = onehot_seq.copy()
#    trajectory_seqs = [first_seq]
#
#    mutations = []
#    scores = []
#
#    for traj_idx in range(n_steps):
#
#        last_seq = trajectory_seqs[-1]
#        last_seq_sparse = np.argwhere(last_seq==1).flatten()
#        
#        log_probs = get_log_probs_single_seq(last_seq, WW, BB, boundaries)
#
#        # Convert to prob
#        probs = np.exp(log_probs)
#
#        ##############################
#        ## Do not pick current again
#        #probs[last_seq_sparse] = 0
#        ##############################
#
#        if LOO is not None:
#            probs[LOO] = 0
#    
#        # Normalize for random.choice
#        probs = probs/np.sum(probs)
#    
#        # Choose
#        mutate_idx = np.random.choice(n_bits, p=probs)
#        
#        mutations.append(mutate_idx)
#        scores.append(log_probs[mutate_idx])
#
#        # Mutate
#        mutate_vidx = VIDX_map[mutate_idx]
#        last_seq_sparse[mutate_vidx] = mutate_idx
#    
#        next_seq = np.zeros(n_bits, dtype=int)
#        next_seq[last_seq_sparse] = 1
#    
#        trajectory_seqs.append(next_seq)
#
#    return first_seq, mutations, scores



def mutate_single_tempered(onehot_seq, beta, n_steps, WW, BB, boundaries, VIDX_map, n_bits, LOO=None):

    first_seq = onehot_seq.copy()
    trajectory_seqs = [first_seq]

    mutations = []
    scores = []

    for traj_idx in range(n_steps):

        last_seq = trajectory_seqs[-1]
        last_seq_sparse = np.argwhere(last_seq==1).flatten()
        
        log_probs = get_log_probs_single_seq(last_seq, WW, BB, boundaries)

        log_probs_tempered = beta*log_probs

        # Convert to prob
        probs = np.exp(log_probs_tempered)

        # Do not pick LOO if specified
        if LOO is not None:
            probs[LOO] = 0

        ###############################
        ## Do not pick current again
        #probs[last_seq_sparse] = 0
        ###############################

        # Normalize for random.choice
        probs = probs/np.sum(probs)
    
        # Choose
        mutate_idx = np.random.choice(n_bits, p=probs)
        
        mutations.append(mutate_idx)
        scores.append(log_probs[mutate_idx])

        # Mutate
        mutate_vidx = VIDX_map[mutate_idx]
        last_seq_sparse[mutate_vidx] = mutate_idx
    
        next_seq = np.zeros(n_bits, dtype=int)
        next_seq[last_seq_sparse] = 1
    
        trajectory_seqs.append(next_seq)

    return first_seq, mutations, scores

#################################################################










#################################################################

def get_trajectory_onehot(first_seq_onehot, mutations, VIDX_map, n_bits):

    trajectory_seqs = [first_seq_onehot]

    for mutate_idx in mutations:
        
        last_seq = trajectory_seqs[-1]
        last_seq_sparse = np.argwhere(last_seq==1).flatten()

        mutate_vidx = VIDX_map[mutate_idx]
        last_seq_sparse[mutate_vidx] = mutate_idx

        next_seq = np.zeros(n_bits, dtype=int)
        next_seq[last_seq_sparse] = 1

        trajectory_seqs.append(next_seq)


    

    return np.vstack(trajectory_seqs)



    

