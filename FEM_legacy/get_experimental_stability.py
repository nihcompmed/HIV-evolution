import numpy as np
import pickle
import helper_functions_v2 as hf
import scipy

#################################

# Load consensus

# From https://hivdb.stanford.edu/pages/documentPage/consensus_amino_acid_sequences.html
PR_consensus = [*'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF']

PR_consensus_arr = np.array(PR_consensus)

#################################

# Load onehot dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

#################################

# Load FEM for no treatment
fname = 'FEM_prot_evolution_None.p'
dbfile = open(fname, 'rb')
FEM_dict = pickle.load(dbfile)
dbfile.close()

WW = FEM_dict['WW']
BB = FEM_dict['BB']

#################################



dirr = 'flynn_2017paper_data'


tm_data = 'SI_data1_Tm.txt'

fname = open(f'{dirr}/{tm_data}', 'r')

def single(fname):

    ff = open(fname, 'r')
    
    header = ff.readline()
    
    info_dict = []
    
    data_regress = []
    
    for line in ff:
    
        parser = line.strip('\n').split(',')
    
        dataset = parser[2]
    
        mutations = parser[5].split('/')
    
        mut_seq = PR_consensus_arr.copy()
    
        err_flag = 0
    
        for mut in mutations:
    
            original_AA = mut[0]
            pos = int(mut[1:-1])
            mut_AA = mut[-1]
    
            idx = pos - 1
    
            # sanity check
            if PR_consensus_arr[idx] != original_AA:
                err_flag = 1
                break
    
            mut_seq[idx] = mut_AA
        
        if err_flag:
            continue
    
        try:
            val = float(parser[-2])
        except:
            print(line)
    
        sparse_seq = hf.seq_to_sparse(mut_seq, onehot_dict['varying_pos'], onehot_dict['encode_dict'])
        onehot_seq = hf.seq_to_onehot(mut_seq, onehot_dict)
    
        log_probs = hf.get_log_probs_single_seq(onehot_seq, WW, BB, onehot_dict['boundaries'])
    
        current_log_probs = log_probs[sparse_seq]
    
        m_val = np.amin(current_log_probs)
    
        print(val, m_val)
    
        data_regress.append([val, m_val])
    
        info_dict.append([dataset, mutations, val])
    
    
    data_regress = np.array(data_regress)
    
    result = scipy.stats.spearmanr(data_regress[:,0], data_regress[:,1])
    
    print(result.statistic, result.pvalue)
    
    ff.close()


tm_data = 'SI_data1_Tm.txt'

fname = f'{dirr}/{tm_data}'

single(fname)

rc_data = 'SI_data1_RC.txt'

fname = f'{dirr}/{rc_data}'

single(fname)

#
#header = ff.readline()
#
#info_dict['RC'] = []
#
#for line in ff:
#
#    parser = line.strip('\n').split(',')
#
#    dataset = parser[2]
#
#    mutations = parser[5]
#
#    val = parser[-2]
#
#    info_dict['RC'].append([dataset, mutations, val])
#
#
#ff.close()
#
#print(info_dict)
