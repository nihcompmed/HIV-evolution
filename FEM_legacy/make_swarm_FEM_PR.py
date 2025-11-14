import pickle
from tqdm import tqdm
import sys


# Load fams
dbfile = open(f'PI_treatments_dict.p', 'rb')
do_fams = pickle.load(dbfile)
dbfile.close()


swarm_ff = open('swarm_FEM_PR.sh', 'w')

for treatment in do_fams:
    cmd = f'python3 get_FEM_PR.py {treatment}\n'
    swarm_ff.write(cmd)

swarm_ff.close()


