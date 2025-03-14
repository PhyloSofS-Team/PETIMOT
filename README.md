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

1. Download resources from [Figshare](https://figshare.com/s/ab400d852b4669a83b64):
- Download `default_2025-02-07_21-54-02_epoch_33.pt` into the `weights/` directory
- Download and extract `ground_truth.zip` into the `ground_truth/` directory

2. Run inference and evaluation:
```bash
python -m petimot infer_and_evaluate \
    --model-path weights/default_2025-02-07_21-54-02_epoch_33.pt \
    --list-path eval_list.txt \
    --ground-truth-path ground_truth/ \
    --prediction-path predictions/ \
    --evaluation-path evaluation/
```

### Compare with baseline methods

1. Download baseline predictions from [Figshare](https://figshare.com/s/ab400d852b4669a83b64) :
- Download and extract `baseline_predictions.zip` into the `baselines/` directory

2. Run evaluation:
```bash
python -m petimot evaluate \
    --prediction-path baselines/alphaflow_pdb_distilled/ \
    --ground-truth-path ground_truth/ \
    --output-path evaluation/
```

Available baseline predictions:
- AlphaFlow (distilled)
- ESMFlow (distilled)
- Normal Mode Analysis



### Predict motions for your own PDB files

```bash
# Single PDB structure
python -m petimot infer \
    --model-path weights/default_2025-02-07_21-54-02_epoch_33.pt \
    --list-path protein.pdb \
    --output-path predictions/

# Multiple structures (provide paths in a text file)
python -m petimot infer \
    --model-path weights/default_2025-02-07_21-54-02_epoch_33.pt \
    --list-path protein_list.txt \
    --output-path predictions/
```
