# Scripts for pre- and post-pocessing of PETIMOT: Protein Motion Inference from Sparse Data

## Script 1: create_traj.py

Generate a trajectory by deforming a starting 3D structure along a mode (linear motion) given as input.

By default, it will generate 50 frames, the mode can optionnally be applied in the opposite direction. 

```bash
python create_traj.py ground_truth/1MF2H_1OTSC.pdb predictions/1MF2H_1OTSC_mode_0.txt traj_1MF2H_1OTSC_mode_0.pdb
```

## Script 2: sample_from_predictions.py

Generate a trajectory by deforming a starting 3D structure along a set of modes. The amplitudes are uniformly sampled within a hypersphere of total energy whose diameter depends on the size of the protein (number of residues). 

How to call the function:

```python
results = load_and_sample(
    pt_path="ground_truth/"+prot+".pt",
    pred_modes_dir="predictions/"+prot,
    outPDB="traj_fixed_energy/ensemble_"+prot+".pdb",
    mulFac=1
```

## Script 3: compute_RMSD_vs_GTconf.py

Compute metrics to evaluate the quality of generated ensembles.
These include the RMSD values to known experimental conformations, the coverage (fraction of experimental conformations approximated with deviation smaller than 2.5 Angstroms), the diversity, the precision, and the recall.

How to call the function:

```python
results = compute_ensemble_vs_gt_rmsds(
    ensemble_pdb="traj_fixed_energy/ensemble_petimot_1MF2H_1OTSC.pdb",
    gt_dir="ground_truth/",
    prot_name="1MF2H",
    query_name="1OTSC",
```

## Other scripts

The other scripts are useful to convert from PT to TXT format, and to perform training splits and training sample identifier matching.



