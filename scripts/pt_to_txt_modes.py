#!/usr/bin/env python3
"""
Script pour convertir des fichiers .pt contenant les coordonnées du backbone
et la séquence en fichiers PDB.
"""

import torch
import os
import argparse
from pathlib import Path


def load_pt_file(pt_path):
    """Charge un fichier .pt et extrait les données nécessaires."""
    try:
        data = torch.load(pt_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Erreur lors du chargement de {pt_path}: {e}")
        return None


def process_single_file(pt_path, output_dir=None):
    """Traite un seul fichier .pt."""
    data = load_pt_file(pt_path)
    if data is None:
        return False
    
    # Extraction des données
    modes = data.get('eigvects')
    name = data.get('name', Path(pt_path).stem)
    
    if modes is None:
        print(f"Données manquantes dans {pt_path}")
        return False
    
    # Définir le chemin de sortie
    if output_dir is None:
        output_dir = Path(pt_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    prot_name = Path(pt_path).stem  
    N = modes.shape[1]
    L = modes.shape[0] // 3
    modes_reshaped = modes.view(L, 3, N)

    # Écrire un fichier par mode
    for mode_idx in range(N):
        output_path = output_dir / f"{prot_name}_mode_{mode_idx}.txt"
        with open(output_path, "w") as f:
            for atom_idx in range(L):
                x, y, z = modes_reshaped[atom_idx, :, mode_idx].tolist()
                f.write(f"{x} {y} {z}\n")
    print("Done!")

    return True


def process_directory(input_dir, output_dir=None):
    """Traite tous les fichiers .pt dans un répertoire."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Le répertoire {input_dir} n'existe pas")
        return
    
    pt_files = list(input_path.glob("*.pt"))
    
    if not pt_files:
        print(f"Aucun fichier .pt trouvé dans {input_dir}")
        return
    
    print(f"Traitement de {len(pt_files)} fichiers .pt...")
    
    success_count = 0
    for pt_file in pt_files:
        if process_single_file(pt_file, output_dir):
            success_count += 1
    
    print(f"Traitement terminé: {success_count}/{len(pt_files)} fichiers convertis")


def main():
    parser = argparse.ArgumentParser(description="Convertir des fichiers .pt en fichiers TXT contenant les modes")
    parser.add_argument("input", help="Fichier .pt ou répertoire contenant des fichiers .pt")
    parser.add_argument("-o", "--output", help="Répertoire de sortie (par défaut: même répertoire que l'entrée)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix == '.pt':
        # Traiter un seul fichier
        process_single_file(args.input, args.output)
    elif input_path.is_dir():
        # Traiter un répertoire
        process_directory(args.input, args.output)
    else:
        print(f"Erreur: {args.input} n'est ni un fichier .pt ni un répertoire valide")


if __name__ == "__main__":
    main()