import numpy as np
import pickle

# From https://hivdb.stanford.edu/pages/documentPage/consensus_amino_acid_sequences.html
PR_consensus = [*'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF']

fname = 'PR.txt'

ff = open(fname, 'r')

header = ff.readline()

parser = header.strip('\n').split('\t')

print(parser[7:106])

# As advised by elife paper
valid_symbs = [\
            'A' ,'C' ,'D' ,'E' ,'F'\
            ,'G' ,'H' ,'I' ,'K' ,'L'\
            ,'M' ,'N' ,'P' ,'Q' ,'R'\
            ,'S' ,'T' ,'V' ,'W' ,'Y'\
            , '*'\
            ]

skip = 0

treatment_dict = dict()

for line in ff:

    parser = line.strip('\n').split('\t')

    treatment = parser[6]

    if treatment not in treatment_dict:
        treatment_dict[treatment] = []

    this_seq = PR_consensus[:]

    flag = 1

    for p_idx, aa in enumerate(parser[7:106]):

        if aa != '-':
            if aa not in valid_symbs:
                flag = 0
                break
            this_seq[p_idx] = aa

    if not flag:
        skip += 1
        continue

    treatment_dict[treatment].append(this_seq)

ff.close()

print(f'Skipped seqs {skip}')

min_thresh = 100

do_fams = dict()


for treatment in treatment_dict:

    if len(treatment_dict[treatment]) == 0:
        continue

    treatment_dict[treatment] = np.unique(np.vstack(treatment_dict[treatment]), axis=0)

    if treatment_dict[treatment].shape[0] >= min_thresh:
        treats = treatment.split(',')
        # Ignore Unknown and PI
        if 'Unknown' in treats or 'PI' in treats:
            continue

        do_fams[treatment] = treatment_dict[treatment]


dbfile = open(f'PI_treatments_dict.p', 'wb')
pickle.dump(do_fams, dbfile)
dbfile.close()


