from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from joblib import Parallel, delayed



fname = '../preprocess_data/PI_treatments_dict.p'
dbfile = open(fname, 'rb')
PI_treatments = pickle.load(dbfile)
dbfile.close()



for treatment in PI_treatments:

    seqs = PI_treatments[treatment]

    encoder = OneHotEncoder()

    trained_encoder = encoder.fit(seqs)

    info = dict()
    info['onehot_encoder'] = trained_encoder
    info['logistic_regression'] = dict()


    onehot_seqs = np.array(trained_encoder.transform(seqs).todense())

    lenns = []

    for cat in trained_encoder.categories_:

        lenns.append(len(cat))

    lenns = np.array(lenns)

    boundaries = np.hstack([[0], np.cumsum(lenns)])

    def single_pos(pos):

        if lenns[pos] == 1:
            return pos, None

        targets = seqs[:,pos]

        start = boundaries[pos]
        end = boundaries[pos+1]

        XX = np.hstack([onehot_seqs[:, :start], onehot_seqs[:, end:]])

        # Initialize and train the model
        model = LogisticRegression(penalty='l2', max_iter=10000)
        model.fit(XX, targets)

        return pos, model

    result = Parallel(n_jobs=24, verbose=12)\
                (delayed(single_pos)(pos) for pos in range(len(boundaries)-1))

    for rr in result:

        pos = rr[0]
        model = rr[1]

        if model is None:
            continue

        info['logistic_regression'][pos] = model

    fname = f'models/treatment{treatment}_PR_evol_onehot_logistic.p'
    dbfile = open(fname, 'wb')
    pickle.dump(info, dbfile)
    dbfile.close()



