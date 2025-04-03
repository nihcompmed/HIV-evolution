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

swarm_ff = open('swarm_stay_drug_res_LOO.sh', 'w')


for LOO in range(onehot_dict['n_bits']):

    fname = f'LOO_res_drug_res_stats/prob_to_reach_drug_res_and_stay_LOO{LOO}.p'
    if os.path.isfile(fname):
        continue

    cmd = f'python3 get_trajs_stay_drug_res_LOO.py {LOO}\n'
    swarm_ff.write(cmd)


swarm_ff.close()
