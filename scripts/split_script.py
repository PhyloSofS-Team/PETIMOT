import random
from collections import defaultdict

def split_families(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split families into train/val/test (70/15/15), but for val and test,
    take only the first unique identifier per family.
    
    Args:
        input_file: path to file containing identifiers
        train_ratio: proportion of families for training (0.7 = 70%)
        val_ratio: proportion of families for validation (0.15 = 15%)
        test_ratio: proportion of families for testing (0.15 = 15%)
        seed: random seed for reproducibility
    """
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Dictionary to group identifiers by family
    families = defaultdict(list)
    
    # Read file and group by family
    with open(input_file, 'r') as f:
        for line in f:
            identifier = line.strip()
            if identifier:  # Skip empty lines
                # Extract family (part before first underscore)
                family = identifier.split('_')[0]
                families[family].append(identifier)
    
    # Convert to list of families for shuffling
    family_list = list(families.keys())
    random.shuffle(family_list)
    
    # Calculate split sizes
    total_families = len(family_list)
    train_size = int(total_families * train_ratio)
    val_size = int(total_families * val_ratio)
    # test_size will be the remainder
    
    # Split families
    train_families = family_list[:train_size]
    val_families = family_list[train_size:train_size + val_size]
    test_families = family_list[train_size + val_size:]
    
    # Create identifier lists for each split
    train_ids = []
    val_ids = []
    test_ids = []
    
    # Train: take all identifiers from train families
    for family in train_families:
        train_ids.extend(families[family])
    
    # Val: take only the first identifier from each val family
    for family in val_families:
        val_ids.append(families[family][0])  # First identifier only
        
    # Test: take only the first identifier from each test family
    for family in test_families:
        test_ids.append(families[family][0])  # First identifier only
    
    # Save files
    with open('train_list.txt', 'w') as f:
        for identifier in train_ids:
            f.write(identifier + '\n')
    
    with open('val_list.txt', 'w') as f:
        for identifier in val_ids:
            f.write(identifier + '\n')
    
    with open('test_list.txt', 'w') as f:
        for identifier in test_ids:
            f.write(identifier + '\n')
    
    # Print statistics
    print(f"Total families: {total_families}")
    print(f"Total identifiers in input: {sum(len(ids) for ids in families.values())}")
    print()
    print(f"Train: {len(train_families)} families ({len(train_families)/total_families*100:.1f}%), {len(train_ids)} identifiers")
    print(f"Val:   {len(val_families)} families ({len(val_families)/total_families*100:.1f}%), {len(val_ids)} identifiers (1 per family)")
    print(f"Test:  {len(test_families)} families ({len(test_families)/total_families*100:.1f}%), {len(test_ids)} identifiers (1 per family)")
    print()
    print(f"Total identifiers used: {len(train_ids) + len(val_ids) + len(test_ids)}")
    print("Files created: train_list.txt, val_list.txt, test_list.txt")
    
    # Show some example families in each split
    print("\nExample families:")
    print(f"Train: {train_families[:5]}...")
    print(f"Val: {val_families[:3]}...")
    print(f"Test: {test_families[:3]}...")
    
    # Show some examples of what goes into val/test
    if val_families:
        print(f"\nExample val entries (first ID per family):")
        for i, family in enumerate(val_families[:3]):
            print(f"  Family {family}: {families[family][0]} (from {len(families[family])} available)")
    
    if test_families:
        print(f"\nExample test entries (first ID per family):")
        for i, family in enumerate(test_families[:3]):
            print(f"  Family {family}: {families[family][0]} (from {len(families[family])} available)")

if __name__ == "__main__":
    # Run the script
    split_families('full_list.txt')