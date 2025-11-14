import pickle



# Load fams
dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()


swarm_ff = open('swarm_drug_res_stats_rare_events.sh', 'w')


for background_treatment in do_fams:

    for beta in [0.4, 0.6, 0.8, 1.0]:

        cmd = f'python3 get_trajs_drug_res_rare_events.py {background_treatment} {beta}\n'
        swarm_ff.write(cmd)


swarm_ff.close()
