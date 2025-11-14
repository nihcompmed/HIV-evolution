import numpy as np
from Bio.PDB import PDBList, PDBParser, MMCIFParser
from scipy.spatial import distance_matrix
import os
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

def get_hiv_protease_distance_matrix_1hhp(pdb_id="1HHP", chain_id="A", n_residues=99):
    """
    Downloads and parses the HIV Protease PDB file (1HHP), extracts
    C-alpha coordinates from Chain A only, and computes the distance matrix.
    
    Requires: numpy, biopython, scipy
    """
    
    # Use .cif format
    pdb_file = f"{pdb_id.lower()}.cif"
    pdbl = PDBList()
    
    # Download the file if it doesn't exist
    if not os.path.exists(pdb_file):
        print(f"Downloading {pdb_id} from RCSB PDB (as mmCIF)...")
        try:
            # This will download '1hhp.cif'
            pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format='mmCif')
        except Exception as e:
            print(f"Error downloading PDB file: {e}")
            print("Attempting to fall back to PDB format...")
            try:
                pdb_file = f"pdb{pdb_id.lower()}.ent" # Legacy PDB format name
                if not os.path.exists(pdb_file):
                    pdbl.retrieve_pdb_file(pdb_id, pdir='.', file_format='pdb')
            except Exception as e2:
                print(f"Fallback to PDB format also failed: {e2}")
                return None
            
    if not os.path.exists(pdb_file):
         # Check for mmCIF default name if PDB fallback was used/failed
        if os.path.exists(f"{pdb_id.lower()}.cif"):
            pdb_file = f"{pdb_id.lower()}.cif"
        else:
            print(f"Error: PDB file {pdb_file} not found after download attempt.")
            return None

    print(f"Parsing structure from {pdb_file}...")
    
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        
        # *** FIX 1: Use explicit MMCIFParser for .cif files ***
        if pdb_file.endswith(".cif"):
            parser = MMCIFParser() 
        else:
            parser = PDBParser()
            
        try:
            structure = parser.get_structure(pdb_id, pdb_file)
        except Exception as e:
            print(f"Error parsing structure: {e}")
            return None

    try:
        # *** FIX 2: Access first model robustly via iterator ***
        model = next(structure.get_models())
        
        if chain_id not in model:
            print(f"Error: Chain {chain_id} not found in model.")
            available_chains = [c.id for c in model]
            print(f"Available chains: {available_chains}")
            if not available_chains:
                print("No chains found in model.")
                return None
            chain_id = available_chains[0]
            print(f"Warning: Using first available chain: {chain_id}")
            
        chain = model[chain_id]
    except StopIteration:
        print(f"Error: No models found in the structure object for {pdb_file}.")
        return None
    except Exception as e:
        print(f"Error accessing model or chain: {e}")
        return None

    # Extract C-alpha coordinates for all 99 residues
    coords = np.zeros((n_residues, 3))
    
    valid_residues = 0
    for i, res_id in enumerate(range(1, n_residues + 1)):
        try:
            residue = chain[res_id]
            ca_atom = residue['CA']
            coords[i, :] = ca_atom.get_coord()
            valid_residues += 1
        except KeyError:
            print(f"Warning: Residue {res_id} or its CA atom not found in chain {chain_id}. Setting coords to NaN.")
            coords[i, :] = np.nan
            
    print(f"Successfully extracted {valid_residues} C-alpha coordinates out of {n_residues}.")

    # Calculate intra-chain distance matrix
    dist_matrix = distance_matrix(coords, coords)
    
    # Save the matrix
    output_file = "hiv_protease_1HHP_A_dist_matrix.npy"
    np.save(output_file, dist_matrix)
    print(f"\nSuccessfully computed and saved {n_residues}x{n_residues} distance matrix to:")
    print(f"{os.path.abspath(output_file)}")
    
    return dist_matrix

if __name__ == "__main__":
    # Ensure you have biopython, numpy, and scipy installed
    # pip install biopython numpy scipy
    get_hiv_protease_distance_matrix_1hhp(pdb_id="1HHP", chain_id="A", n_residues=99)
