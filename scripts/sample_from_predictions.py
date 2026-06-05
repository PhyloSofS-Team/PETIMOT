import sys
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

AA_1TO3 = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}


def sample_fixed_energy(modes_pred, n_residues, n_samples, energy_per_residue=1.0, mulFac=100):
    """
    Randomly draws amplitude for a fixed total energy (scaling with protein size)
    """
    n_modes = modes_pred.shape[1]
    total_energy = energy_per_residue * n_residues  # scale avec la taille
    
    samples = []
    for _ in range(n_samples):
        # Coefficients aléatoires sur la sphère de rayon sqrt(total_energy)
        raw = np.random.randn(n_modes)
        # Normaliser pour que sum(coeffs²) = total_energy
        coeffs = raw / np.linalg.norm(raw) * np.sqrt(total_energy) * mulFac
        displacement = modes_pred @ coeffs
        new_coords = coords_flat + displacement
        samples.append(new_coords.reshape(n_residues, 3))
    
    return np.stack(samples)
    
def load_and_sample(pt_path, pred_modes_dir, outPDB="", n_modes_pred=4, mulFac=50, n_samples=50, energy_per_residue=1.0):
   
    # 1. Load data associated with the reference structure
    data = torch.load(pt_path)
    bb = data['bb']  # (1, N_residues, 4, 3) ou (N_residues, 4, 3)
    if bb.dim() == 4:
        bb = bb.squeeze(0)  # (N_residues, 4, 3)  
    n_residues = bb.shape[0]
    
    # Extraire seulement les CA (index 1)
    coords_ca = bb[:, 1, :]  # (N_residues, 3)
    # get the sequence
    seq = data['seq']
    
    # 2. Load predicted motions
    modes_pred = []
    for i in range(n_modes_pred):
        mode = np.loadtxt(f"{pred_modes_dir}_mode_{i}.txt")  # (N_residues, 3)
        modes_pred.append(mode.flatten())  # (N_residues * 3,)
    modes_pred = np.stack(modes_pred, axis=1)  # (N_residues * 3, n_modes_pred)
    # Normalise each motion (column)
    modes_pred = modes_pred / np.linalg.norm(modes_pred, axis=0, keepdims=True)
    
    # 3. Sample
    coords_flat = coords_ca.flatten().numpy()  # (N_residues * 3,)
    
    samples = []

    if outPDB:
        fOUTPDB = open(outPDB,"w")

    # estimate total energy based on protein size
    total_energy = energy_per_residue * n_residues  

    # generate conformational ensemble
    for sample_idx in range(n_samples):
        # compute coefficients
        raw = np.random.randn(n_modes_pred)
        coeffs = raw / np.linalg.norm(raw) * np.sqrt(total_energy) * mulFac
        # deform starting structure
        displacement = modes_pred @ coeffs
        new_coords = coords_flat + displacement
        samples.append(new_coords.reshape(n_residues, 3))
    
        # write ensemble in a multi-PDB
        if outPDB:
            fOUTPDB.write(f"MODEL {sample_idx + 1}\n")
            new_coords = new_coords.reshape(n_residues, 3)
            for i, coord in enumerate(new_coords):
                res3 = AA_1TO3.get(seq[i], 'UNK')
                # res = seq[i] if i < len(seq) else 'X'
                # Convertir 1-lettre en 3-lettres si besoin
                fOUTPDB.write(
                    f"ATOM  {i+1:5d}  CA  {res3} A{i+1:4d}    "
                    # f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n"
                    )
            fOUTPDB.write("ENDMDL\n")
    if outPDB:
        fOUTPDB.close()
    
    samples = np.stack(samples)  # (n_samples, N_residues, 3)
    
    # 4. Compute amplitude and RMSF
    mean_structure = samples.mean(axis=0)
    rmsf_pred = np.sqrt(((samples - mean_structure) ** 2).sum(axis=2).mean(axis=0))
    displacements = samples - coords_ca.numpy()  # (n_samples, N, 3)
    norms = np.linalg.norm(displacements.reshape(n_samples, -1), axis=1)
    print(f"Amplitude moyenne: {norms.mean():.2f} Å")
    print(f"RMSF moyen: {rmsf_pred.mean():.2f} Å")
    
    return {
        'rmsf': rmsf_pred,
        'amplitudes': norms,
    }


if __name__ == '__main__':
    
    # list of samples given as input 
    fnam = sys.argv[1]
    fIN = open(fnam,"r")
    lines = fIN.readlines()
    fIN.close()

    for line in lines:
        prot = line[:-1]
        print(prot)
        results = load_and_sample(
            pt_path="ground_truth/"+prot+".pt",
            pred_modes_dir="predictions/"+prot,
            outPDB="traj_fixed_energy/ensemble_"+prot+".pdb",
            mulFac=1
            )

