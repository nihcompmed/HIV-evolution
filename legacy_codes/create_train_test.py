import pickle
import numpy as np
import random
import math

dbfile = open('PI_drug_res_dict.p', 'rb')
drug_res_dict = pickle.load(dbfile)
dbfile.close()

def get_train_test(mat, test_split):

    nn = mat.shape[0]

    train_mask = np.ones(nn, dtype=int)

    n_test = math.floor(nn * 0.2)

    test_idxs = np.random.choice(nn, size=n_test, replace=False)

    train_mask[test_idxs] = 0

    return train_mask

    

test_split = 0.2

train_test_drug_res = dict()

for drug in drug_res_dict:

    train_test_drug_res[drug] = dict()
    train_test_drug_res[drug]['train'] = dict()
    train_test_drug_res[drug]['test'] = dict()

    mat = drug_res_dict[drug]['low_res']
    train_low_res_mask = get_train_test(mat, test_split)
    train_idxs = np.argwhere(train_low_res_mask == 1).flatten()
    test_idxs = np.argwhere(train_low_res_mask == 0).flatten()

    train_test_drug_res[drug]['train']['low_res'] = mat[train_idxs]
    train_test_drug_res[drug]['test']['low_res'] = mat[test_idxs]

    
    mat = drug_res_dict[drug]['high_res']
    train_high_res_mask = get_train_test(mat, test_split)
    train_idxs = np.argwhere(train_high_res_mask == 1).flatten()
    test_idxs = np.argwhere(train_high_res_mask == 0).flatten()

    train_test_drug_res[drug]['train']['high_res'] = mat[train_idxs]
    train_test_drug_res[drug]['test']['high_res'] = mat[test_idxs]


dbfile = open('PI_drug_res_train_test_dict.p', 'wb')
pickle.dump(train_test_drug_res, dbfile)
dbfile.close()

