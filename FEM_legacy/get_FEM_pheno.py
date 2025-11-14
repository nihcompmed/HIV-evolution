import helper_functions_v2 as hf
import pickle
import numpy as np
import FEM
import scipy
import sklearn
from sklearn import metrics




dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()


dbfile = open('PI_drug_res_train_test_dict.p', 'rb')
train_test_drug_res = pickle.load(dbfile)
dbfile.close()

for drug in ['FPV','ATV','IDV','LPV','NFV','SQV']:

    all_xx = []
    all_yy = []
    
    high_res = train_test_drug_res[drug]['train']['high_res']
    all_xx.append(high_res)
    all_yy.append([1]*high_res.shape[0])
    
    low_res = train_test_drug_res[drug]['train']['low_res']
    all_xx.append(low_res)
    all_yy.append([0]*low_res.shape[0])
    
    all_xx = np.vstack(all_xx)
    all_yy = np.hstack(all_yy)
    
    all_xx_onehot = []
    
    for seq in all_xx:
        onehot_seq = hf.seq_to_onehot(seq, onehot_dict)
        all_xx_onehot.append(onehot_seq)
    
    all_xx_onehot = np.vstack(all_xx_onehot)
    
    
    H0, W = FEM.FEM_fit(all_xx_onehot,all_yy,niter_max=100,l2=100)
    
    save_fname = f'FEM_drugres_{drug}.npz'
    print(f'Saving {save_fname}...')
    np.savez(save_fname, weights=W, bias=H0)

