#read a pdb file and a mode file and outputs a trajectory file

import numpy as np
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Read a PDB file and a mode file and outputs a trajectory file.')
parser.add_argument('pdb_file', help='Path to the PDB file.')
parser.add_argument('mode_file', help='Path to the mode file.')
parser.add_argument('output_file', help='Path to the output trajectory file.')
parser.add_argument('--num_modes', type=int, default=15, help='Number of times the mode is applied (default: 15).')
parser.add_argument('--num_frames', type=int, default=50, help='Number of frames (default: 50).')
parser.add_argument('--invert', type=int, default=0, help='inversion of the mode (default: 0).')

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
pdb_file = args.pdb_file
mode_file = args.mode_file
output_file = args.output_file
num_modes = args.num_modes
num_frames = args.num_frames
invert = args.invert

# read the pdb file
# read only the CA atoms
list_of_coordinates = []
list_of_residues = []
with open(pdb_file, 'r') as pdb:
    # dont read all the lines just the CA atoms of the first model
    # a lign beginning with ENDMDL should end the reading
    for line in pdb:
        if line.startswith('ENDMDL'):
            break
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            parsed_line = [
                    line[i:j] for i, j in [(0, 6), (6, 11), (12, 16), (17, 20), (21, 22), (22, 26), (30, 38), (38, 46), (46, 54)]
                ]
            coordinates = parsed_line[6:9]
            # convert the coordinates to float
            coordinates = [float(i) for i in coordinates]
            list_of_coordinates.append(coordinates)
            residue = parsed_line[3]
            list_of_residues.append(residue)

np_coords = np.array(list_of_coordinates)
len_coords = len(np_coords)
# read the mode file

np_mode = np.loadtxt(mode_file)
if invert == 1:
    np_mode = - np_mode

# only read the first len_coords rows
np_mode = np_mode[:len_coords]
# assert that the number of residues in the pdb file is the same as the number of rows in the mode file

assert len(np_coords) == len(np_mode)

# compute the trajectory

# for each mode, add the mode to the coordinates
lin = np.linspace(0, num_modes, num_frames)
np_traj = np.einsum('i,ijk->ijk', lin, np_mode[None,:]) + np_coords[None,:,:]

# write the trajectory to a file
with open(output_file, 'w') as traj:
    for i in range(len(np_traj)):
        for j in range(len(np_traj[i])):
            traj.write('ATOM  {:5d}  CA  {:3s} A{:4d}    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00           C  \n'.format(
                j+1, list_of_residues[j], j+1, np_traj[i][j][0], np_traj[i][j][1], np_traj[i][j][2]
            ))
        traj.write('ENDMDL\n')