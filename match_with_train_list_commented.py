#!/usr/bin/env python3
# Author: Elodie Laine
# Usage: python script.py
# Purpose: This script performs train-validation-evaluation splits based on sequence ID clustering 
#          for the petimot tool, ensuring non-redundant datasets with controlled sequence similarity.

import sys
import random


def getQueries(fname):
    """
    Parse a file containing protein pairs and create a dictionary mapping proteins to their queries.
    
    Args:
        fname (str): Path to input file with format "protein_query" on each line
        
    Returns:
        dict: Dictionary where keys are proteins and values are lists of queries
    """
    # Open and read the file
    with open(fname, "r") as fIN:
        lines = fIN.readlines()

    # Create the dictionary
    d = {}
    for line in lines:
        # Split each line by underscore to get protein and query
        prots = line.strip().split("_")
        if prots[0] not in d:
            d[prots[0]] = []
        d[prots[0]].append(prots[1])

    return d


def getMatchHigherLevel(fnameDB, dEval, dTrainVal):
    """
    Match higher-level clusters with their corresponding lower-level collections.
    
    Args:
        fnameDB (str): Path to the cluster database file
        dEval (dict): Dictionary of evaluation set proteins and their queries
        dTrainVal (dict): Dictionary of training and validation sets proteins and their queries
        
    Returns:
        dict: Dictionary where keys are higher-level cluster representatives and values are tuples of 
              ([train-val proteins], [eval proteins]) that belong to that cluster
    """
    # Open and read the database file
    with open(fnameDB, "r") as fIN:
        lines = fIN.readlines()

    d = {}
    # Each line represents a cluster of PDB chains
    for line in lines:
        # Get individual chains in the cluster
        prots = line.strip().split()
        
        # Use the first protein as the cluster representative
        d[prots[0]] = ([], [])  # Initialize tuple with empty lists for train-val and eval proteins
        
        # For each protein in the cluster
        for p in prots:
            # If it belongs to the train or validation set
            if p in dTrainVal:
                d[prots[0]][0].append(p)
            # If it belongs to the evaluation set
            if p in dEval:
                d[prots[0]][1].append(p)
                
    return d


def splitHigherLevel(dEns, random_seed=42):
    """
    Split higher-level clusters into training, validation, and evaluation sets.
    
    Args:
        dEns (dict): Dictionary mapping higher-level clusters to their proteins
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (dtrainval, lval, ltrain, deval_strict) where:
               - dtrainval: dictionary of clusters with proteins in train/val sets but not in eval
               - lval: list of clusters selected for validation
               - ltrain: list of clusters selected for training
               - deval_strict: dictionary of clusters with proteins only in eval set
    """
    random.seed(random_seed)
    
    # Clusters that have at least one protein in any set
    dall = {k: v[0] for k, v in dEns.items() if (len(v[1]) + len(v[0])) > 0}
    
    # Clusters with proteins only in train/val sets (not in eval)
    dtrainval = {k: v[0] for k, v in dEns.items() if len(v[1]) == 0 and len(v[0]) > 0}
    
    # Clusters with proteins only in eval set (not in train/val)
    deval_strict = {k: v[1] for k, v in dEns.items() if len(v[1]) > 0 and len(v[0]) == 0}
    
    # Clusters with at least one protein in eval set (may overlap with train/val)
    deval_relax = {k: v[1] for k, v in dEns.items() if len(v[1]) > 0}
    
    # Calculate number of validation samples (10% of total clusters)
    n_val = int((len(dtrainval) + len(deval_strict)) / 10)
    
    # Randomly sample clusters for validation
    ival = random.sample(range(len(dtrainval)), n_val)
    myKeys = list(dtrainval.keys())
    
    # Create lists of cluster IDs for validation and training
    lval = [myKeys[i] for i in ival]
    ltrain = [myKeys[i] for i in range(len(myKeys)) if i not in ival]
    
    # Print statistics about the split
    print("Number of higher-order collections with members in train, val or eval:", len(dall))
    print("Number of higher-order collections with members in train or val and nothing from eval:", len(dtrainval))
    print("Number of higher-order collections with members in eval:", len(deval_relax))
    print("Number of higher-order collections with members in eval and nothing from train-val:", len(deval_strict))
    print("Sample validation clusters:", lval[1:5])
    print("Number of higher-order collections in the new validation set:", len(lval))
    print("Sample training clusters:", ltrain[1:5])
    print("Number of higher-order collections in the new training set:", len(ltrain))
    
    return dtrainval, lval, ltrain, deval_strict


def reduceRedundancyInEval(dEns, dEval, random_seed=42):
    """
    Reduce redundancy in the evaluation set by selecting one protein per cluster.
    
    Args:
        dEns (dict): Dictionary mapping higher-level clusters to their proteins
        dEval (dict): Dictionary of evaluation set proteins and their queries
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary of selected evaluation proteins and their queries
    """
    random.seed(random_seed)
    
    # Get clusters with proteins in eval set
    deval_relax = {k: v[1] for k, v in dEns.items() if len(v[1]) > 0}
    
    # Create new evaluation dictionary with reduced redundancy
    deval = {}
    for k in deval_relax:
        # Randomly select one protein from each cluster
        myColl = random.sample(deval_relax[k], 1)[0]
        
        # Check if the selected protein has multiple queries (warning case)
        if len(dEval[myColl]) > 1:
            print("warning!!", myColl, dEval[myColl])
            
        # Add only the first query for the selected protein
        deval[myColl] = [dEval[myColl][0]]

    return deval


def sampleLowerLevel(dtrainval, myL, myD, random_seed=42):
    """
    Sample queries for lower-level collections based on a stratified approach.
    
    Args:
        dtrainval (dict): Dictionary of training and validation proteins
        myL (list): List of higher-level clusters to process
        myD (dict): Dictionary mapping proteins to their queries
        random_seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary of selected proteins and their queries
    """
    random.seed(random_seed)
    
    dres = {}
    # Sampling strategy: number of samples to take based on available proteins
    # For 1 protein, take 5 queries; for 2 proteins, take 3 and 2 queries; etc.
    n_samples = ([5], [3, 2], [2, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1])
    
    for higherColl in myL:
        # Get lower-level collections for this cluster
        lowerColl = dtrainval[higherColl]
        n = len(lowerColl)
        p = min(n, 5)  # Cap at 5 proteins per cluster
        
        # If we have 5 or more proteins, randomly sample 5
        if p == 5:
            selectedColl = random.sample(lowerColl, p)
        else:
            # Otherwise, use all available proteins
            selectedColl = lowerColl
            
        # Get the sampling distribution for this number of proteins
        nbs = n_samples[p-1]
        
        # Sample queries for each selected protein
        for i in range(p):
            j = nbs[i]  # Number of queries to sample for this protein
            dres[selectedColl[i]] = myD[selectedColl[i]][:j]
            
    return dres


def writeDico(dEns):
    """
    Write cluster matching information to a CSV file.
    
    Args:
        dEns (dict): Dictionary of clusters and their proteins
    """
    with open("match_eval.csv", "w") as fOUT:
        for k in dEns:
            n = len(dEns[k])
            if n > 0:
                fOUT.write(k + "," + str(n) + "," + "-".join(dEns[k]) + "\n")


def write_queries(d, fname):
    """
    Write protein-query pairs to a file.
    
    Args:
        d (dict): Dictionary mapping proteins to their queries
        fname (str): Output file name
    """
    with open(fname, "w") as fOUT:
        for k in d:
            for q in d[k]:
                fOUT.write(k + "_" + q + "\n")


def write_values(d, fname):
    """
    Write only query values to a file.
    
    Args:
        d (dict): Dictionary mapping proteins to their queries
        fname (str): Output file name
    """
    with open(fname, "w") as fOUT:
        for k in d:
            for q in d[k]:
                fOUT.write(q + "\n")


if __name__ == "__main__":
    # Load evaluation set data
    evalF = "eval_list.txt"
    dEval = getQueries(evalF)
    
    # Load training and validation sets data
    trvalF = "train_val_list.txt"
    dTrainVal = getQueries(trvalF)
    
    # Load full training set data
    trF = "full_train_list.txt"
    dTrain = getQueries(trF)
    
    # Match the collections using 30% sequence identity, 80% coverage threshold
    dEns = getMatchHigherLevel("rewrited_clusterDB_30_80.tsv", dEval, dTrainVal)
    
    # Split into training, validation and evaluation sets
    dtrainval, lval, ltrain, deval_strict = splitHigherLevel(dEns)
    
    # Sample validation queries and write to file
    dval = sampleLowerLevel(dtrainval, lval, dTrainVal)
    write_queries(dval, "val_nr_list_12_05.txt")
    
    # Sample training queries and write to file
    dtrain = sampleLowerLevel(dtrainval, ltrain, dTrainVal)
    write_queries(dtrain, "train_nr_list_12_05.txt")
    
    # Reduce redundancy in evaluation set and write to file
    deval = reduceRedundancyInEval(dEns, dEval)
    write_queries(deval, "eval_nr_list_12_05.txt")
    
    # Print number of strict evaluation clusters
    print("Number of strict evaluation clusters:", len(deval_strict))
    
    # Generate strict evaluation set based on full training
    dEns2 = getMatchHigherLevel("rewrited_clusterDB_30_80.tsv", deval, dTrain)
    dtrainval, lval, ltrain, deval_strict = splitHigherLevel(dEns2)
    write_values(deval_strict, "eval_strict_list_12_05.txt")
    
    # Generate even stricter evaluation set based on train+val
    dEns3 = getMatchHigherLevel("rewrited_clusterDB_30_80.tsv", deval, dTrainVal)
    dtrainval, lval, ltrain, deval_strict = splitHigherLevel(dEns3)
    write_values(deval_strict, "eval_even_stricter_list_12_05.txt")
