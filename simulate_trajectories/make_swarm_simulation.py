import pickle
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Worker Function ---
# This function processes a single parameter combination. It will be run in parallel.
def check_and_generate_command(params):
    """
    Checks if a job for the given parameters is complete. If not, generates the command.

    Args:
        params (tuple): A tuple containing (treatment, beta, LOO, n_bits).

    Returns:
        str or None: The command string if the job needs to run, otherwise None.
    """
    treatment, beta, LOO, n_bits = params
    
    # Construct the expected output filename
    if LOO < n_bits:
        fname = f'results_1000/simualtion_res_treatment{treatment}_beta{beta}_LOO{LOO}.p'
    else:
        fname = f'results_1000/simualtion_res_treatment{treatment}_beta{beta}_LOO_None.p'
    
    # Use the robust check to see if the file exists and is a valid pickle file
    job_is_complete = False
    if os.path.isfile(fname):
        try:
            # An empty file will have a size of 0. A corrupted pickle will fail to load.
            if os.path.getsize(fname) > 0:
                with open(fname, 'rb') as p_file:
                    pickle.load(p_file)
                job_is_complete = True
        except (pickle.UnpicklingError, EOFError):
            # File is corrupted or empty, so we treat it as incomplete.
            job_is_complete = False
            
    if job_is_complete:
        return None
    else:
        return f'python3 simulate_PR_trajectories.py {treatment} {beta} {LOO}\n'

# --- Main Script Logic ---

if __name__ == "__main__":
    betas = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]

    fname = 'PI_treatments_dict.p'
    with open(fname, 'rb') as dbfile:
        PI_treatments = pickle.load(dbfile)

    # 1. Generate all parameter combinations first
    all_params = []
    print("Generating and preparing all parameter combinations...")
    for treatment in PI_treatments:
        fname = f'treatment{treatment}_PR_evol_onehot_logistic.p'
        with open(fname, 'rb') as dbfile:
            info = pickle.load(dbfile)
        
        PR_onehot_encoder = info['onehot_encoder']
        
        bit_idx = 0
        do_LOO = []
        for cat in PR_onehot_encoder.categories_:
            for _ in cat:
                if len(cat) != 1 and treatment != 'None':
                    do_LOO.append(bit_idx)
                bit_idx += 1
        n_bits = bit_idx
        do_LOO.append(n_bits)  # For the 'None' case

        for LOO in do_LOO:
            for beta in betas:
                all_params.append((treatment, beta, LOO, n_bits))
    
    print(f"Total parameter combinations to check: {len(all_params)}")

    # 2. Process all combinations in parallel with a progress bar
    print("Checking for completed jobs across all available CPU cores...")
    # n_jobs=24 uses all available cores.
    # The 'tqdm' wrapper provides the progress bar.
    commands_to_run = Parallel(n_jobs=24)(
        delayed(check_and_generate_command)(params) for params in tqdm(all_params)
    )

    # 3. Filter out the None results (for jobs that are already complete)
    final_commands = [cmd for cmd in commands_to_run if cmd is not None]

    # 4. Write the necessary commands to the swarm file
    with open('swarm_simulate.sh', 'w') as swarm_ff:
        swarm_ff.writelines(final_commands)

    print("-" * 50)
    print(f"Finished. Found {len(final_commands)} jobs to run.")
    print("The 'swarm_simulate.sh' file has been generated.")
