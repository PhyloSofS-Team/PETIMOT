# PETIMOT: Protein Motion Inference from Sparse Data

PETIMOT (Protein sEquence and sTructure-based Inference of MOTions) predicts protein conformational changes using SE(3)-equivariant graph neural networks and pre-trained protein language models.

## Installation

```bash
# Create and activate conda environment
conda create -n petimot python=3.9
conda activate petimot

# Clone and install
git clone https://github.com/PhyloSofS-Team/PETIMOT.git
cd petimot
pip install -r requirements.txt
```


## Usage

### Reproduce paper results 

1. Download resources from [Figshare](https://figshare.com/s/fe86b5dd8ad0985e2fab):
- Download `default.pt` into the `model_weights/` directory
- Download and extract `ground_truth.tgz` into the `ground_truth/` directory

2. Run inference and evaluation:
```bash
python -m petimot infer_and_evaluate \
    --model-path model_weights/default.pt \
    --list-path splits/test824.txt \
    --ground-truth-path ground_truth/ \
    --prediction-path predictions/ \
    --evaluation-path evaluation/
```

### Compare with baseline methods

1. Download baseline predictions from [Figshare](https://figshare.com/s/fe86b5dd8ad0985e2fab) :
- Download and extract `baseline_predictions.tgz` into the `baseline_predictions/` directory

2. Run evaluation:
```bash
python -m petimot evaluate \
    --prediction-path baseline_predictions/alphaflow_pdb_distilled/ \
    --ground-truth-path ground_truth/ \
    --output-path evaluation/
```

Available baseline predictions:
- BioEmu
- AlphaFlow (distilled)
- ESMFlow (distilled)
- Normal Mode Analysis (various flavours of the Elastic Network Model)

### Predict motions for your own PDB files

```bash
# Single PDB structure
python -m petimot infer \
    --model-path weights/default.pt \
    --list-path protein.pdb \
    --output-path predictions/

# Multiple structures (provide paths in a text file)
python -m petimot infer \
    --model-path weights/default.pt \
    --list-path protein_list.txt \
    --output-path predictions/
```

### Generate a conformational ensemble

1. Download predictions from [Figshare](https://figshare.com/s/fe86b5dd8ad0985e2fab) :
- Download and extract `predictions.tgz` into the `predictions/` directory
- Create `traj_fixed_energy` directory

2. Generate:
```bash
python scripts/sample_from_predictions.py test824.txt
```
