import numpy as np
import os
import torch
from pathlib import Path
from Bio.PDB import PDBParser, Superimposer
from Bio.Align import PairwiseAligner
import MDAnalysis as mda
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import sys

def align_sequences(seq1: str, seq2: str):
    from Bio.Align import PairwiseAligner
    
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    
    alignments = aligner.align(seq1, seq2)
    best = alignments[0]

    # DEBUG
   # print(f"seq1 length: {len(seq1)}")
   # print(f"seq2 length: {len(seq2)}")
   # print(f"Alignment:\n{best}")
    
    aln_str = str(best).split('\n')

    # best.aligned donne [(start, end)] pour chaque segment continu sans gap
    # On veut tous les indices des positions alignées
    idx1, idx2 = [], []
    
    # Parcourir les segments alignés simultanément
    for (start1, end1), (start2, end2) in zip(best.aligned[0], best.aligned[1]):
        # Ces segments ont la même longueur et sont alignés
        length = end1 - start1
        assert end2 - start2 == length  # vérification
        
        for offset in range(length):
            idx1.append(start1 + offset)
            idx2.append(start2 + offset)
    
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    
    assert idx1.max() < len(seq1), f"idx1 out of bounds: {idx1.max()} >= {len(seq1)}"
    assert idx2.max() < len(seq2), f"idx2 out of bounds: {idx2.max()} >= {len(seq2)}"
    
    return idx1, idx2

def compute_lddt(coords1: np.ndarray, coords2: np.ndarray, 
                 thresholds=[0.5, 1.0, 2.0, 4.0], radius=15.0):
    """
    Calcule le lDDT entre deux structures.
    
    Args:
        coords1, coords2: (N, 3) coordonnées des Cα
        thresholds: seuils de distance en Å
        radius: rayon d'inclusion pour les paires locales
    """
    n_atoms = len(coords1)
    
    # Distances dans la structure 1 (référence)
    dist1 = np.linalg.norm(coords1[:, None] - coords1[None, :], axis=2)
    
    # Distances dans la structure 2 (prédiction)
    dist2 = np.linalg.norm(coords2[:, None] - coords2[None, :], axis=2)
    
    # Différences de distances
    dist_diff = np.abs(dist1 - dist2)
    
    # Masque pour paires locales (distance < radius dans structure de référence)
    local_mask = (dist1 < radius) & (dist1 > 0)  # exclure diagonale
    
    if not local_mask.any():
        return 0.0
    
    # Compter combien de paires respectent chaque seuil
    scores = []
    for threshold in thresholds:
        preserved = (dist_diff[local_mask] < threshold).sum()
        total = local_mask.sum()
        scores.append(preserved / total)
    
    # lDDT = moyenne sur tous les seuils
    return np.mean(scores)

def compute_rmsd_after_alignment(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Superpose coords1 sur coords2 et retourne le RMSD."""
    # Centrer
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    
    # Kabsch algorithm
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Corriger réflexion si nécessaire
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    c1_aligned = c1 @ R
    rmsd = np.sqrt(((c1_aligned - c2) ** 2).sum(axis=1).mean())
    return rmsd

def compute_ensemble_vs_gt_rmsds(
    ensemble_pdb: str,
    gt_dir: str,
    prot_name: str,  # e.g., "3F59A"
    query_name: str,  # e.g., "3F59D" pour la protéine de référence
    compute_lddt_scores: bool = True, 
    cluster_gt: bool = True,
    max_gt_per_cluster: int = 1,  # nombre de représentants par cluster
    rmsd_threshold: float = 2.0,  # seuil pour considérer deux conf comme distinctes
):
    """
    Calcule les RMSD entre un ensemble et toutes les conformations GT d'une protéine.
    
    Args:
        ensemble_pdb: chemin vers le multi-PDB
        gt_dir: dossier contenant les .pt GT
        prot_name: nom de la protéine (e.g., "3F59A")
        query_name: nom de la conformation de référence (pour récupérer la séquence)
        cluster_gt: si True, ne garde que des conformations GT diverses
        max_gt_per_cluster: nombre max de GT par cluster
        rmsd_threshold: seuil RMSD pour clustering (Å)
    """
    import glob
    
    # 1. Trouver tous les .pt qui commencent par prot_name
    pattern = os.path.join(gt_dir, f"{prot_name}_*.pt")
    all_gt_pts = sorted(glob.glob(pattern))
    
    if not all_gt_pts:
        raise ValueError(f"No GT files found matching {pattern}")
    
    #print(f"Found {len(gt_pts)} GT conformations for {prot_name}")
    
    # 2. Récupérer la séquence de référence
    query_pt = os.path.join(gt_dir, f"{prot_name}_{query_name}.pt")
    if not os.path.exists(query_pt):
        raise ValueError(f"Query file not found: {query_pt}")
    
    query_data = torch.load(query_pt)
    ensemble_seq = query_data['seq']

    # Dans compute_ensemble_vs_gt_rmsds, ajoute après avoir chargé query_data:
    ensemble_seq = query_data['seq']
    #print(f"Séquence length: {len(ensemble_seq)}")

    # 3. Charger toutes les conformations GT
    all_gt_coords = []
    all_gt_seqs = []
    all_gt_names = []
    
    for pt_path in all_gt_pts:
        data = torch.load(pt_path)
        bb = data['bb']
        if bb.dim() == 4:
            bb = bb.squeeze(0)
        coords_ca = bb[:, 1, :].numpy()
        all_gt_coords.append(coords_ca)
        all_gt_seqs.append(data['seq'])
        all_gt_names.append(Path(pt_path).stem)

    # 4. Clustering des GT si demandé
    if cluster_gt and len(all_gt_coords) > 1:
        # Calculer matrice de RMSD pairwise entre toutes les GT
        n_gt = len(all_gt_coords)
        rmsd_matrix = np.zeros((n_gt, n_gt))
        
        for i in range(n_gt):
            for j in range(i+1, n_gt):
                # Aligner les séquences
                idx_i, idx_j = align_sequences(all_gt_seqs[i], all_gt_seqs[j])
                coords_i = all_gt_coords[i][idx_i]
                coords_j = all_gt_coords[j][idx_j]
                
                # RMSD après superposition
                rmsd = compute_rmsd_after_alignment(coords_i, coords_j)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        # Clustering hiérarchique
        condensed_dist = squareform(rmsd_matrix)
        Z = linkage(condensed_dist, method='average')
        clusters = fcluster(Z, t=rmsd_threshold, criterion='distance')
        
        # Sélectionner des représentants
        selected_indices = []
        for cluster_id in np.unique(clusters):
            cluster_members = np.where(clusters == cluster_id)[0]
            
            # Prendre jusqu'à max_gt_per_cluster par cluster
            # Stratégie : prendre ceux avec le plus petit RMSD moyen vers les autres du cluster
            if len(cluster_members) <= max_gt_per_cluster:
                selected_indices.extend(cluster_members)
            else:
                # Prendre les plus "centraux"
                centrality = rmsd_matrix[cluster_members][:, cluster_members].mean(axis=1)
                most_central = cluster_members[np.argsort(centrality)[:max_gt_per_cluster]]
                selected_indices.extend(most_central)
        
        # Filtrer les GT
        gt_coords_list = [all_gt_coords[i] for i in selected_indices]
        gt_seqs = [all_gt_seqs[i] for i in selected_indices]
        gt_names = [all_gt_names[i] for i in selected_indices]
        
        print(f"Clustered {n_gt} GT into {len(np.unique(clusters))} groups, kept {len(selected_indices)} conformations")
    else:
        gt_coords_list = all_gt_coords
        gt_seqs = all_gt_seqs
        gt_names = all_gt_names
    
    # 5. Charger l'ensemble
    u = mda.Universe(ensemble_pdb)
    n_frames = len(u.trajectory)
    
    # 6. Charger les conformations GT
   # gt_coords_list = []
   # gt_seqs = []
    #gt_names = []
    
    # remove the query from the GT for evaluation
    #gt_pts.remove(query_pt)
    # go over the gt
    #for pt_path in gt_pts:
    #    data = torch.load(pt_path)
    #    bb = data['bb']
    #    if bb.dim() == 4:
    #        bb = bb.squeeze(0)
    #    coords_ca = bb[:, 1, :].numpy()
    #    gt_coords_list.append(coords_ca)
    #    gt_seqs.append(data['seq'])
    #    gt_names.append(Path(pt_path).stem)
    
    # 6. Calculer les RMSD et lDDT
    all_rmsds = np.zeros((n_frames, len(gt_coords_list)))
    all_lddts = np.zeros((n_frames, len(gt_coords_list))) if compute_lddt_scores else None
    
    # align to the ensemble
    d_ali = {}
    for gt_idx, (gt_coords, gt_seq) in enumerate(zip(gt_coords_list, gt_seqs)):
        idx_ens, idx_gt = align_sequences(ensemble_seq, gt_seq)
        d_ali[gt_idx] = (idx_ens, idx_gt)

    # pour chaque frame de la trajectoire
    for frame_idx, ts in enumerate(u.trajectory):
        # récupère les coordonnées
        frame_coords = u.select_atoms("name CA").positions
        
        # pour chaque conformation GT
        for gt_idx, (gt_coords, gt_seq) in enumerate(zip(gt_coords_list, gt_seqs)):

          #  frame_coords = u.select_atoms("name CA").positions
           # print(f"Nombre de CA dans le PDB: {len(frame_coords)}")
           # print(f"Séquence du PDB: {len(ensemble_seq)}")

            # Aligner les séquences
            #idx_ens, idx_gt = align_sequences(ensemble_seq, gt_seq)

            #print(f"frame_coords shape: {frame_coords.shape}")
            #print(f"idx_ens range: {idx_ens.min()} to {idx_ens.max()}")
            #print(f"idx_gt range: {idx_gt.min()} to {idx_gt.max()}")
            #print(f"ensemble_seq length: {len(ensemble_seq)}")
            #print(f"gt_seq length: {len(gt_seq)}")
            
            # Extraire les coords alignées
            coords_ens_aligned = frame_coords[d_ali[gt_idx][0]]
            coords_gt_aligned = gt_coords[d_ali[gt_idx][1]]
            
            # Calculer RMSD entre la conf générée et la conf GT
            rmsd = compute_rmsd_after_alignment(coords_ens_aligned, coords_gt_aligned)
            all_rmsds[frame_idx, gt_idx] = rmsd

            if compute_lddt_scores:
                lddt = compute_lddt(coords_ens_aligned, coords_gt_aligned)
                all_lddts[frame_idx, gt_idx] = lddt
    
    # 6. Métriques
    min_rmsd_per_frame = all_rmsds.min(axis=1)
    best_gt_per_frame = all_rmsds.argmin(axis=1)  # quel GT est le plus proche pour chaque frame

    min_rmsd_per_gt = all_rmsds.min(axis=0)
    best_frame_per_gt = all_rmsds.argmin(axis=0)  # quelle frame est la plus proche pour chaque GT

    # Vérifier si certaines frames dominent
    from collections import Counter
    frame_counts = Counter(best_frame_per_gt)
    most_common_frame, count = frame_counts.most_common(1)[0]

    results = {
        'all_rmsds': all_rmsds,
        'min_rmsd_per_frame': min_rmsd_per_frame,
        'best_gt_per_frame': best_gt_per_frame,  
        'min_rmsd_per_gt': min_rmsd_per_gt,
        'best_frame_per_gt': best_frame_per_gt,  
        'mean_min_rmsd_to_gt': min_rmsd_per_gt.mean(), 
        'mean_min_rmsd_per_frame': min_rmsd_per_frame.mean(),
        'min_min_rmsd_to_gt': min_rmsd_per_gt.min(), 
        'min_min_rmsd_per_frame': min_rmsd_per_frame.min(),
        'max_min_rmsd_to_gt': min_rmsd_per_gt.max(), 
        'max_min_rmsd_per_frame': min_rmsd_per_frame.max(),
        'gt_names': gt_names,
        'n_frames': n_frames,
        'n_gt': len(gt_coords_list),
        'frame_coverage_stats': {
            'unique_frames_used': len(set(best_frame_per_gt)),
            'most_common_frame': int(most_common_frame),
            'most_common_frame_count': int(count),
            }
        }

    if compute_lddt_scores:
        results.update({
            'all_lddts': all_lddts,
            'max_lddt_per_frame': all_lddts.max(axis=1),
            'max_lddt_per_gt': all_lddts.max(axis=0),
            'mean_max_lddt_to_gt': all_lddts.max(axis=0).mean(),
            'mean_max_lddt_per_frame': all_lddts.max(axis=1).mean(),
        })

    return results


def compute_ensemble_vs_ensemble_metrics(
    pred_pdb: str,
    ref_pdb: str,
    compute_lddt_scores: bool = True,
):
    """
    Compare deux ensembles conformationnels (même séquence, pas d'alignement nécessaire).
    
    Args:
        pred_pdb: multi-PDB de l'ensemble prédit
        ref_pdb: multi-PDB de l'ensemble de référence
        compute_lddt_scores: calculer aussi les lDDT
        
    Returns:
        dict avec métriques RMSD, lDDT, recall, precision
    """
    # 1. Charger les deux ensembles
    u_pred = mda.Universe(pred_pdb)
    u_ref = mda.Universe(ref_pdb)
    
    n_frames_pred = len(u_pred.trajectory)
    n_frames_ref = len(u_ref.trajectory)
    
    # Vérifier que les séquences matchent
    n_atoms_pred = len(u_pred.select_atoms("name CA"))
    n_atoms_ref = len(u_ref.select_atoms("name CA"))
    
    if n_atoms_pred != n_atoms_ref:
        raise ValueError(f"Mismatch: pred has {n_atoms_pred} CA, ref has {n_atoms_ref} CA")
    
    print(f"Comparing {n_frames_pred} pred frames vs {n_frames_ref} ref frames")
    
    # 2. Extraire toutes les coordonnées
    pred_coords = []
    for ts in u_pred.trajectory:
        pred_coords.append(u_pred.select_atoms("name CA").positions.copy())
    pred_coords = np.array(pred_coords)  # (n_frames_pred, n_atoms, 3)
    
    ref_coords = []
    for ts in u_ref.trajectory:
        ref_coords.append(u_ref.select_atoms("name CA").positions.copy())
    ref_coords = np.array(ref_coords)  # (n_frames_ref, n_atoms, 3)
    
    # 3. Calculer matrice RMSD (pred vs ref)
    all_rmsds = np.zeros((n_frames_pred, n_frames_ref))
    all_lddts = np.zeros((n_frames_pred, n_frames_ref)) if compute_lddt_scores else None
    
    for i in range(n_frames_pred):
        for j in range(n_frames_ref):
            rmsd = compute_rmsd_after_alignment(pred_coords[i], ref_coords[j])
            all_rmsds[i, j] = rmsd
            
            if compute_lddt_scores:
                lddt = compute_lddt(pred_coords[i], ref_coords[j])
                all_lddts[i, j] = lddt
    
    # 4. Métriques
    # Pour chaque frame prédite, quelle est la ref la plus proche ?
    min_rmsd_per_pred = all_rmsds.min(axis=1)  # (n_frames_pred,)
    best_ref_per_pred = all_rmsds.argmin(axis=1)
    
    # Pour chaque frame ref, quelle est la pred la plus proche ?
    min_rmsd_per_ref = all_rmsds.min(axis=0)  # (n_frames_ref,)
    best_pred_per_ref = all_rmsds.argmin(axis=0)
    
    # Diversité : combien de frames distinctes couvrent les refs ?
    from collections import Counter
    pred_counts = Counter(best_pred_per_ref)
    most_common_pred, count = pred_counts.most_common(1)[0] if pred_counts else (0, 0)
    
    results = {
        'all_rmsds': all_rmsds,
        
        # Precision : les prédictions sont-elles proches de refs ?
        'min_rmsd_per_pred': min_rmsd_per_pred,
        'best_ref_per_pred': best_ref_per_pred,
        'mean_min_rmsd_per_pred': min_rmsd_per_pred.mean(),  # RMSD moyen des preds vers leur meilleure ref
        
        # Recall : les refs sont-elles couvertes par les preds ?
        'min_rmsd_per_ref': min_rmsd_per_ref,
        'best_pred_per_ref': best_pred_per_ref,
        'mean_min_rmsd_per_ref': min_rmsd_per_ref.mean(),  # RMSD moyen des refs vers leur meilleure pred
        
        'n_frames_pred': n_frames_pred,
        'n_frames_ref': n_frames_ref,
        
        'coverage_stats': {
            'unique_preds_used': len(set(best_pred_per_ref)),
            'most_common_pred': int(most_common_pred),
            'most_common_pred_count': int(count),
        }
    }
    
    if compute_lddt_scores:
        max_lddt_per_pred = all_lddts.max(axis=1)
        max_lddt_per_ref = all_lddts.max(axis=0)
        
        results.update({
            'all_lddts': all_lddts,
            'max_lddt_per_pred': max_lddt_per_pred,
            'max_lddt_per_ref': max_lddt_per_ref,
            'precision_lddt': max_lddt_per_pred.mean(),
            'recall_lddt': max_lddt_per_ref.mean(),
        })
    
    return results


if __name__ == '__main__':
    
    # list of samples given as input 
    fnam = sys.argv[1]
    fIN = open(fnam,"r")
    lines = fIN.readlines()
    fIN.close()

    # give as argument the type of analysis
    # exp when computing the metrics with respect to the experimental structures (ground truth)
    # simul when computing the metrics with respect to the structures generated with the ground truth modes 
    typeAnal = "exp" # sys.argv[1]

    if typeAnal == "exp":
        fOUT = open("test.csv","w")
        fOUT.write("prot,n_gt,n_frames_used,mean_min_rmsd_to_gt,min_min_rmsd_to_gt,max_min_rmsd_to_gt,recall,precision\n")

    # deprecated
    #if typeAnal == "simul":
    #    fOUT = open("rmsd_lddt_test_pm_50_simul.csv","w")
    #    fOUT.write("prot,mean_min_rmsd_per_pred,mean_min_rmsd_per_ref,recall,precision\n")

    for line in lines:
        prot = line[:-1]
        print(prot)
        if typeAnal == "exp":
            results = compute_ensemble_vs_gt_rmsds(
                ensemble_pdb="trajectories/traj_fixed_energy/ensemble_petimot_"+prot+".pdb",
                # alphaflow original "trajectories/traj_alphaflow/"+prot+"_aligned.pdb",
                # alphaflow sampled "trajectories/traj_fixed_energy/ensemble_af_"+prot+".pdb",
                gt_dir="../../../ground_truth/",
                prot_name=prot.split("_")[0],
                query_name=prot.split("_")[1],
                cluster_gt=False,
                )
            fOUT.write(prot+",")
            fOUT.write(",".join([str(results['n_gt']),str(results['frame_coverage_stats']['unique_frames_used']),str(results['mean_min_rmsd_to_gt']),str(results['min_min_rmsd_to_gt']),str(results['max_min_rmsd_to_gt']),str(results['mean_max_lddt_to_gt']),str(results['mean_max_lddt_per_frame'])]))
            fOUT.write("\n")
            print("Tous les RMSD pairwise min par conf GT:")
            print(results['min_rmsd_per_gt'])
            print(results['best_frame_per_gt'])
            print(f"Diversité : {results['frame_coverage_stats']['unique_frames_used']}/{results['n_frames']} frames distinctes couvrent les GT")
            print(f"Coverage des GT (mean min RMSD): {results['mean_min_rmsd_to_gt']:.2f} Å")
            print("Tous les lDDT pairwise max par conf GT:")
            print(results['max_lddt_per_gt'])
            print(f"Recall: {results['mean_max_lddt_to_gt']}")
            print(f"Precision: {results['mean_max_lddt_per_frame']}")
            print(f"GT names: {results['gt_names']}")

        # deprecated
        # if typeAnal == "simul":
            # results = compute_ensemble_vs_ensemble_metrics(
                # pred_pdb="traj2/ensemble_petimot_"+prot+".pdb",
                # ref_pdb="traj2/ensemble_gt_"+prot+".pdb",
                # )
            # fOUT.write(prot+",")
            # fOUT.write(",".join([str(results['mean_min_rmsd_per_pred']),str(results['mean_min_rmsd_per_ref']),str(results['recall_lddt']),str(results['precision_lddt'])]))
            # fOUT.write("\n")

            # print(f"Precision (RMSD): {results['mean_min_rmsd_per_pred']:.2f} Å")
            # print(f"Recall (RMSD): {results['mean_min_rmsd_per_ref']:.2f} Å")
            # print(f"Precision (lDDT): {results['precision_lddt']:.3f}")
            # print(f"Recall (lDDT): {results['recall_lddt']:.3f}")
            # print(f"Coverage: {results['coverage_stats']['unique_preds_used']}/{results['n_frames_pred']} frames used")

    fOUT.close()



# Utilisation
#results = compute_ensemble_vs_gt_rmsds(
#    ensemble_pdb="traj/ensemble_petimot_1MF2H_1OTSC.pdb",
#    gt_dir="../../../ground_truth/",
#    prot_name="1MF2H",
#    query_name="1OTSC",
#)



