# Scripts for pre- and post-pocessing of PETIMOT: Protein Motion Inference from Sparse Data

## Script 1: compute_RMSD_vs_GTconf.py

Compute metrics to evaluate the quality of generated ensembles.
These include the RMSD values to known experimental conformations, the coverage (fraction of experimental conformations approximated with deviation smaller than 2.5 Angstroms), the diversity, the precision, and the recall

```python
results = compute_ensemble_vs_gt_rmsds(
    ensemble_pdb="traj_fixed_energy/ensemble_petimot_1MF2H_1OTSC.pdb",
    gt_dir="ground_truth/",
    prot_name="1MF2H",
    query_name="1OTSC",
```

