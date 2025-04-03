import pickle
import numpy as np

# From https://hivdb.stanford.edu/pages/documentPage/consensus_amino_acid_sequences.html
PR_consensus = [*'PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF']

# As advised by elife paper
valid_symbs = [\
            'A' ,'C' ,'D' ,'E' ,'F'\
            ,'G' ,'H' ,'I' ,'K' ,'L'\
            ,'M' ,'N' ,'P' ,'Q' ,'R'\
            ,'S' ,'T' ,'V' ,'W' ,'Y'\
            , '*'\
            ]

#for pos, aa in enumerate(PR_consensus):
#    print(pos, aa)

fname = 'PI_DataSet.txt'

ff = open(fname, 'r')

header = ff.readline()

parser = header.strip('\n').split('\t')

seq_dict = dict()

ignore_seqs = 0
consensus_ambi = 0

all_seqs = []

seq_counts = 0

for line in ff:

    seq_counts += 1
    
    parser = line.strip('\n').split('\t')
    this_seq_id = parser[0]
    this_mutations = parser[-1].split(',')

    this_mutations = [x.strip(' ') for x in this_mutations]


    this_seq = PR_consensus[:]

    flag = 1

    for mut in this_mutations:

        C_AA = mut[0]
        M_AA = mut[-1]

        if M_AA not in valid_symbs:
            flag = 0
            break

        try:
            pos = int(mut[1:-1])
        except:
            flag = 0
            break

        if this_seq[pos-1] != C_AA:
            consensus_ambi += 1
            #print(f'current mismatch for seq {this_seq_id} at position {pos}')
            #print(f'consensus is {this_seq[pos-1]} and mutation is {mut}')

        this_seq[pos-1] = M_AA

    if not flag:
        ignore_seqs += 1
        continue

    seq_dict[this_seq_id] = this_seq

    all_seqs.append(this_seq)

print(f'Total seqs:{seq_counts}')

print(f'ignored seqs:{ignore_seqs}')
print(consensus_ambi)

all_seqs = np.vstack(all_seqs)
print(all_seqs.shape)

unique_seqs = np.unique(all_seqs, axis=0)
print(unique_seqs.shape)


dbfile = open('PI_valid_seq_ids.p', 'wb')
pickle.dump(seq_dict, dbfile)
dbfile.close()


