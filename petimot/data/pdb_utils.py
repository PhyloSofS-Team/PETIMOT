from typing import Dict, Tuple, List
import torch
from pathlib import Path
import os


def load_backbone_coordinates(
    pdb_path: str,
    allow_hetatm: bool = False,
) -> Tuple[torch.Tensor, ...]:

    THREE_TO_ONE = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
        "SEC": "U",
        "PYL": "O",
        "UNK": "X",
        # Non-standard and modified amino acids
        "ABA": "A",
        "ALY": "K",
        "BFD": "D",
        "CAF": "C",
        "CAS": "C",
        "CGU": "E",
        "CME": "C",
        "CSD": "C",
        "CSO": "C",
        "CSS": "C",
        "CSX": "C",
        "CXM": "M",
        "DAL": "A",
        "DCY": "C",
        "DHA": "S",
        "DLE": "L",
        "DSN": "S",
        "FME": "M",
        "HIC": "H",
        "HYP": "P",
        "IAS": "D",
        "KCX": "K",
        "LLP": "K",
        "M3L": "K",
        "MDO": "A",
        "MEN": "N",
        "MEQ": "Q",
        "MHO": "M",
        "MLE": "L",
        "MLY": "K",
        "MLZ": "K",
        "MSE": "M",
        "MVA": "V",
        "NEP": "H",
        "NLE": "L",
        "OCS": "C",
        "PCA": "E",
        "PHD": "D",
        "PTR": "Y",
        "SAR": "G",
        "SCH": "C",
        "SCY": "C",
        "SEP": "S",
        "SMC": "C",
        "SME": "M",
        "SNC": "C",
        "TPO": "T",
        "TYS": "Y",
        "YCM": "C",
    }

    residues, types, numbers = [], [], []
    current = {"coords": [], "type": None, "num": None}

    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if not (
                    line.startswith("ATOM")
                    or (allow_hetatm and line.startswith("HETATM"))
                ):
                    continue

                atom = line[12:16].strip()
                res_type = line[17:20].strip()

                if res_type == "HOH" or atom not in ["N", "CA", "C", "O"]:
                    continue

                coords = torch.tensor(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )

                current["coords"].append(coords)

                if not current["type"]:
                    current["type"] = res_type
                    current["num"] = int(line[22:26])

                if len(current["coords"]) == 4:
                    residues.append(torch.stack(current["coords"]))
                    types.append(current["type"])
                    numbers.append(current["num"])
                    current = {"coords": [], "type": None, "num": None}

    except FileNotFoundError:
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    except Exception as e:
        raise ValueError(f"Error parsing PDB file: {e}")

    if not residues:
        raise ValueError("No valid backbone atoms found")

    backbone = torch.stack(residues)

    output_path = f"extracted_{Path(pdb_path).name}"
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            atom_num = 1
            res_num = 1
            for res_idx, residue in enumerate(residues):
                res_type = types[res_idx]
                for atom_idx, (atom_name, coords) in enumerate(
                    zip(["N", "CA", "C", "O"], residue)
                ):
                    # Standard PDB format
                    f.write(
                        f"ATOM  {atom_num:5d}  {atom_name:<3s} {res_type:3s} A{res_num:4d}    "
                        f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"
                        f"  1.00  0.00           {atom_name[0]:>2s}\n"
                    )
                    atom_num += 1
                res_num += 1
            f.write("END\n")

    outputs = {"bb": backbone}
    seq = "".join(THREE_TO_ONE.get(t, "X") for t in types)
    outputs["seq"] = seq
    outputs["residue_types"] = types
    outputs["residue_numbers"] = numbers
    return outputs
