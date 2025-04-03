
import helper_functions_v2 as hf
import pickle
import numpy as np
import FEM
import scipy
import sklearn
from sklearn import metrics

dbfile = open('PI_drug_res_train_test_dict.p', 'rb')
train_test_drug_res = pickle.load(dbfile)
dbfile.close()

dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()


for drug in ['FPV','ATV','IDV','LPV','NFV','SQV']:

    fname = f'FEM_drugres_{drug}.npz'
    data = np.load(fname)
    
    W = data['weights']
    H0 = data['bias']
    
    
    # Check accuracy
    
    all_xx = []
    all_yy = []
    
    high_res = train_test_drug_res[drug]['test']['high_res']
    all_xx.append(high_res)
    all_yy.append([1]*high_res.shape[0])
    
    low_res = train_test_drug_res[drug]['test']['low_res']
    all_xx.append(low_res)
    all_yy.append([0]*low_res.shape[0])
    
    all_xx = np.vstack(all_xx)
    all_yy = np.hstack(all_yy)
    
    all_xx_onehot = []
    
    for seq in all_xx:
        onehot_seq = hf.seq_to_onehot(seq, onehot_dict)
        all_xx_onehot.append(onehot_seq)
    
    all_xx_onehot = np.vstack(all_xx_onehot)
    
    HH = np.matmul(all_xx_onehot, W) + H0
    
    pred_true_prob = scipy.special.expit(HH)
    
    preds = (pred_true_prob > 0.5).flatten()
    
    match = np.logical_xor(all_yy, preds)
    
    false_idxs = np.argwhere(match == 1).flatten()
    true_idxs = np.argwhere(match == 0).flatten()
    
    true_pos = np.sum(all_yy[true_idxs])
    
    
    false_preds = all_yy[false_idxs]
    true_preds = all_yy[true_idxs]
    
    false_pos = np.sum(false_preds)
    false_neg = len(false_preds) - false_pos
    
    f1 = (2*true_pos)/(2*true_pos + false_pos + false_neg)
    
    print(f'{drug} F1:{f1}')
