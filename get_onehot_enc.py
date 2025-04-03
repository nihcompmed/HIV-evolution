import helper_functions_v2 as hf
import pickle
import numpy as np


dbfile = open('PI_treatments_dict.p', 'rb')
fams = pickle.load(dbfile)
dbfile.close()

all_seqs = []

for treatment in fams:
    print(f'{treatment} {fams[treatment].shape}')
    all_seqs.append(fams[treatment])

dbfile = open('PI_drug_res_train_test_dict.p', 'rb')
train_test_drug_res = pickle.load(dbfile)
dbfile.close()

for drug in train_test_drug_res:

    all_seqs.append(train_test_drug_res[drug]['train']['high_res'])
    all_seqs.append(train_test_drug_res[drug]['test']['high_res'])
    all_seqs.append(train_test_drug_res[drug]['train']['low_res'])
    all_seqs.append(train_test_drug_res[drug]['test']['low_res'])

all_seqs = np.vstack(all_seqs)

#print(all_seqs.shape)
#
#n_cols = all_seqs.shape[1]
#for pos in range(n_cols):
#    this_col = all_seqs[:, pos]
#    uni  = np.unique(this_col)
#    print(pos, uni)
#
#exit()

onehot_dict = hf.get_onehot_dict(all_seqs)


dbfile = open(f'PR_alldrugs_onehot_dict.p', 'wb')
pickle.dump(onehot_dict, dbfile)
dbfile.close()

