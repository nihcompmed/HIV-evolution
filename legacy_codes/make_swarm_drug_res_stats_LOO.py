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

swarm_ff = open('swarm_drug_res_stats_LOO.sh', 'w')


for background_treatment in do_fams:

    for LOO in range(onehot_dict['n_bits']):

        if LOO not in [279,291]:
            continue

        #for beta in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for beta in [0.4, 0.6, 0.8, 1.0]:

            #fname = f'LOO_res_drug_res_stats/mutations_treat{background_treatment}_prob_to_reach_drug_res_LOO{LOO}_beta{beta}.p'

            #if os.path.isfile(fname):
            #    continue

            cmd = f'python3 get_trajs_drug_res_stats_LOO.py {background_treatment} {beta} {LOO}\n'
            swarm_ff.write(cmd)


swarm_ff.close()
