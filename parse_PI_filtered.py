import pickle
import numpy as np
import scipy

dbfile = open('PI_valid_seq_ids.p', 'rb')
valid_seq_dict = pickle.load(dbfile)
dbfile.close()

# Cutoffs for PI drug fold resistance
drug_cut_offs = dict()

drug_cut_offs['ATV'] = (3,15)
drug_cut_offs['DRV'] = (10,90)
drug_cut_offs['FPV'] = (3,15)
drug_cut_offs['IDV'] = (3,15)
drug_cut_offs['LPV'] = (9,55)
drug_cut_offs['NFV'] = (3,6)
drug_cut_offs['SQV'] = (3,15)
drug_cut_offs['TPV'] = (2,8)


fname = 'PI_DataSet.txt'

ff = open(fname, 'r')

header = ff.readline()

parser = header.strip('\n').split('\t')

header_drugs = ['FPV'\
,'ATV'\
,'IDV'\
,'LPV'\
,'NFV'\
,'SQV'\
,'TPV'\
,'DRV']

print(parser)

drug_res_dict = dict()
for drug in drug_cut_offs:
    drug_res_dict[drug] = dict()
    drug_res_dict[drug]['low_res'] = []
    drug_res_dict[drug]['high_res'] = []

this_res = dict()

for line in ff:

    parser = line.strip('\n').split('\t')

    seqID = parser[0]

    if seqID not in valid_seq_dict:
        continue

    for d_idx, drug in enumerate(header_drugs):
        this_drug_res = parser[d_idx+1]
        if this_drug_res != 'NA':
            this_drug_res = float(this_drug_res)
            if this_drug_res >= drug_cut_offs[drug][1]:
                drug_res_dict[drug]['high_res'].append(seqID)
            else:
                drug_res_dict[drug]['low_res'].append(seqID)

ff.close()

def get_unique_seqs(seq_ids):

    seqs = []
    for seq_id in seq_ids:
        seqs.append(valid_seq_dict[seq_id])

    seqs = np.vstack(seqs)

    unique_seqs = np.unique(seqs, axis=0)

    return unique_seqs


# Remove duplicates seqs
for drug in drug_res_dict:
    drug_res_dict[drug]['high_res'] = get_unique_seqs(drug_res_dict[drug]['high_res'])
    drug_res_dict[drug]['low_res'] = get_unique_seqs(drug_res_dict[drug]['low_res'])


# Check if same sequence has low drug res and high drug res
for drug in drug_res_dict:

    mat1 = drug_res_dict[drug]['high_res']
    mat2 = drug_res_dict[drug]['low_res']

    for row1 in mat1:
        for row2 in mat2:

            cond = row1 == row2

            if np.all(cond):
                print(row1, row2)

for drug in drug_res_dict:
    print(drug, drug_res_dict[drug]['low_res'].shape, drug_res_dict[drug]['high_res'].shape)

dbfile = open('PI_drug_res_dict.p', 'wb')
pickle.dump(drug_res_dict, dbfile)
dbfile.close()



