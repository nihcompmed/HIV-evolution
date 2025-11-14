import pickle
import numpy as np
import helper_functions_v2 as hf
from sklearn.decomposition import PCA


dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()

dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()


all_onehot_seqs = []
all_yy = []


for treatment in do_fams:

    seqs = do_fams[treatment]

    onehot_seqs = []

    for seq in seqs:

        onehot_seq = hf.seq_to_onehot(seq, onehot_dict)
        onehot_seqs.append(onehot_seq)

    onehot_seqs = np.vstack(onehot_seqs)

    all_onehot_seqs.append(onehot_seqs)

    all_yy.append([treatment]*onehot_seqs.shape[0])


all_onehot_seqs = np.vstack(all_onehot_seqs)
all_yy = np.hstack(all_yy)


pca = PCA(n_components=2)

pca.fit(all_onehot_seqs)


dbfile = open(f'PR_PI_PCA.p', 'wb')
pickle.dump(pca, dbfile)
dbfile.close()


