import pickle
import matplotlib.pyplot as plt
import helper_functions_v2 as hf
import numpy as np
        
from tqdm import tqdm

from joblib import Parallel, delayed

xx_max = 10
yy_max = 20

def get_pareto_optimality(info, background_treatment, drug, xx_max, yy_max):

    this_info = info[background_treatment][drug]

    n_vals = len(this_info.keys())

    r_vals = np.zeros(n_vals)
    m_vals = np.zeros(n_vals)

    for idx, seq in enumerate(this_info):
        r_vals[idx] = this_info[seq][0]
        m_vals[idx] = this_info[seq][1]


    # log log
    r_vals = np.log(-r_vals)
    
    m_vals = -m_vals

    r_vals = r_vals/xx_max
    m_vals = m_vals/yy_max

    nn = np.sqrt((r_vals)**2 + (m_vals)**2)
                
    return nn



# Load onehot dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

n_bits = onehot_dict['n_bits']


def single_LOO(test_LOO):

    dbfile = open(f'LOO_res_drug_res_stats/prob_to_reach_drug_res_and_stay_LOO{test_LOO}.p', 'rb')
    info_test_LOO = pickle.load(dbfile)
    dbfile.close()

    all_info_dict = dict()

    for background_treatment in info_test_LOO:

        for drug in info_test_LOO[background_treatment]:
    
            LOO_nn = get_pareto_optimality(info_test_LOO, background_treatment, drug, xx_max, yy_max)

            all_info_dict[(background_treatment, drug)] = LOO_nn
            
    return all_info_dict
    
    
res = Parallel(n_jobs=12, verbose=4)\
        (delayed(single_LOO)(LOO) for LOO in range(n_bits))


fname = 'pareto_optimality_LOO.p'
dbfile = open(fname, 'wb')
pickle.dump(res, dbfile)
dbfile.close()


