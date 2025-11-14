from sklearn.preprocessing import OneHotEncoder
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression



fname = 'PI_treatments_dict.p'
dbfile = open(fname, 'rb')
PI_treatments = pickle.load(dbfile)
dbfile.close()

fname = 'PI_drug_res_dict.p'
dbfile = open(fname, 'rb')
PR_drug_res = pickle.load(dbfile)
dbfile.close()


for drug in PR_drug_res:

    all_seqs = []

    low_res_seqs = PR_drug_res[drug]['low_res']
    high_res_seqs = PR_drug_res[drug]['high_res']

    targets = [0]*low_res_seqs.shape[0] + [1]*high_res_seqs.shape[0]


    for treatment in PI_treatments:

        print(treatment, drug)

        all_seqs = np.vstack([low_res_seqs, high_res_seqs, PI_treatments[treatment]])

        encoder = OneHotEncoder()

        trained_encoder = encoder.fit(all_seqs)

        drug_res_seqs = np.vstack([low_res_seqs, high_res_seqs])

        drug_res_seqs_onehot = trained_encoder.transform(drug_res_seqs)

        # Initialize and train the model
        model = LogisticRegression(penalty='l2', max_iter=10000)
        model.fit(drug_res_seqs_onehot, targets)

        info = dict()
        info['onehot_encoder'] = trained_encoder
        info['logistic_regression'] = model

        fname = f'treatment{treatment}_drugres{drug}_onehot_logistic.p'
        dbfile = open(fname, 'wb')
        pickle.dump(info, dbfile)
        dbfile.close()



