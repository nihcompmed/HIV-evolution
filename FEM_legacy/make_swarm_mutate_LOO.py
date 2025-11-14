import pickle
import os

# Load fams
dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()


# Load onehot_dict
dbfile = open(f'PR_alldrugs_onehot_dict.p', 'rb')
onehot_dict = pickle.load(dbfile)
dbfile.close()

swarm_ff = open('swarm_mutate_LOO.sh', 'w')

n_traj = 1000

for treatment in do_fams:

    for LOO in range(onehot_dict['n_bits']):

        for beta in [0.2, 0.4, 0.6, 0.8, 1.0]:

            fname = f'LOO_results/trajs{n_traj}_{treatment}_LOO{LOO}_beta{beta}.p'

            if os.path.isfile(fname):
                continue

            cmd = f'python3 mutate_seqs_LOO.py {treatment} {beta} {LOO}\n'
            swarm_ff.write(cmd)

swarm_ff.close()
