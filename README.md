Code to generate main figures for manuscript "Forecasting drug resistant HIV protease evolution".

Step 1: Train models

Run Python scripts in `train_models` to train models. Results will be saved to `train_models/models`

Step 2: Simulate trajectories

This is the most compute-intensive job. We provide scripts to run on HPC.\
Run `simulate_trajectories/make_swarm_simulation.py` to generate a bash script `swarm_simulate.sh`.\
Run `swarm_simulate.sh` on HPC to simulate evolution trajectories.\
Script `simulate_trajectories/simulate_PR_trajectories.sh` is the worker that simulates the trajectories and infers drug resistance.\
Run `simulate_trajectories/consolidate_phi.py` to compute phi(tau) for all cases.\

Step 3: Plot main text figures

Each main text figure has a dedicated folder in the parent fodler `plots`. Run plotting scripts available in those folders.

