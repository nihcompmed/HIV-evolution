import numpy as np
import pickle

def parse_ref_info(fname):

    ff = open(fname, 'r')
    # Ignore header
    ff.readline()

    for line in ff:

        line = line.split('\t')
        print(line)
        exit()

    ff.close()

def valid_seq(seq, consensus_aa):
    # As advised by elife paper
    valid_symbs = [\
                'A' ,'C' ,'D' ,'E' ,'F'\
                ,'G' ,'H' ,'I' ,'K' ,'L'\
                ,'M' ,'N' ,'P' ,'Q' ,'R'\
                ,'S' ,'T' ,'V' ,'W' ,'Y'\
                , '*'\
                ]

    # only symbols from valid symbols ( No insertion (#), deletion (~), and no ambiguity)
    for pos, symb in enumerate(seq):
        if symb == '-':
            seq[pos] = consensus_aa[pos]
        if seq[pos] not in valid_symbs:
            return 0, None, seq[pos]

    # No > 1% gaps (.)
    N = len(seq)
    n_gaps = seq.count('.')
    if n_gaps/N > 0.9:
        #print('Skipping because lot of gaps')
        return 0, None, None

    return 1, seq, None

def get_consensus_seq(enz):

    consensus = dict()

    # Consensus from https://hivdb.stanford.edu/pages/documentPage/consensus_amino_acid_sequences.html
    consensus['PR'] =\
            "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

    consensus['RT'] = "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGFTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYAGIKVKQLCKLLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKIATESIVIWGKTPKFKLPIQKETWEAWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTDRGRQKVVSLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDKSESELVSQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL"

    consensus['integrase'] = "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKIILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"

    consensus['capsid'] = "PIVQNLQGQMVHQAISPRTLNAWVKVVEEKAFSPEVIPMFSALSEGATPQDLNTMLNTVGGHQAAMQMLKETINEEAAEWDRLHPVHAGPIAPGQMREPRGSDIAGTTSTLQEQIGWMTNNPPIPVGEIYKRWIILGLNKIVRMYSPTSILDIRQGPKEPFRDYVDRFYKTLRAEQASQEVKNWMTETLLVQNANPDCKTILKALGPAATLEEMMTACQGVGGPGHKARVL"


    consensus_aa = []

    for pos, aa in enumerate(consensus[enz]):
        consensus_aa.append(aa)

    return np.array(consensus_aa)



def parse_raw_info(source, target, enz):


    consensus_aa = get_consensus_seq(enz)
    
    ff = open(source, 'r')

    parsed = dict()

    parsed['treatments'] = dict()

    line = ff.readline()
    line = line.strip('\n')
    line = line.split('\t')

    count = 0

    ignored_symbs = []

    for line in ff:

        line = line.strip('\n')
        line = line.split('\t')

        refid = line[0]
        ptid = line[0]
        isoname = line[2]
        region = line[3]
        year = line[4]
        treatment = line[6]

        seq = line[7:-2]


        flag, seq, symb = valid_seq(seq, consensus_aa)

        if symb not in ignored_symbs:
            ignored_symbs.append(symb)

        if not flag:
            continue

        # Multiple sequences obtained from single patient?

        parsed[count] = {'refid':refid\
                        , 'region':region\
                        , 'PtID':ptid\
                        , 'year':year\
                        , 'seq':np.array(seq)\
                        }
        if treatment not in parsed['treatments']:
            parsed['treatments'][treatment] = []

        parsed['treatments'][treatment].append(count)

        count += 1


    ff.close()

    print(ignored_symbs)

    for treatment in parsed['treatments']:
        print(f"Number of seqs treated with {treatment} is {len(parsed['treatments'][treatment])}")

    dbfile = open(target, 'wb')
    pickle.dump(parsed, dbfile)
    dbfile.close()


# Get hamming distance from consensus        
def get_hamming_from_consensus(seq, enzyme):

    consensus_aa = get_consensus_seq(enzyme)

    ddist = 0

    for pos, aa in enumerate(seq):
        if aa != consensus_aa[pos]:
            ddist += 1
    
    return ddist





