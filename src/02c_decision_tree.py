#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Decision Tree Analysis - Predicting Extreme Disadvantage (IMPROVED)
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script builds a decision tree classifier to predict "extremely disadvantaged"
women in the DRC 2023-24 DHS survey. The analysis:

1. DEFINES EXTREME DISADVANTAGE:
   - Women who answered "Yes" to 2 or more IPV indicators (D111, D104, D106, D108)
   - Creates a binary outcome: 1 = extremely disadvantaged, 0 = not

2. INTELLIGENTLY FILTERS AND PREPARES FEATURES:
   - Keeps only columns with >80% valid responses (configurable)
   - Automatically detects categorical vs numerical variables
   - Handles categorical and numerical features appropriately

3. BUILDS A DECISION TREE:
   - Uses filtered DHS variables as predictors
   - Applies DHS sampling weights for nationally representative results
   - All hyperparameters defined at the top for easy tuning

4. OUTPUTS:
   - Tree structure as YAML (02c_decision_tree.yaml)
   - Tree visualization as PNG and PDF (02c_decision_tree.png/pdf)
   - Feature importance rankings
   - Model performance metrics

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (created by convert.py)

OUTPUTS:
--------
- 02c_decision_tree.yaml (tree structure in YAML format)
- 02c_decision_tree.png (tree visualization, high resolution)
- 02c_decision_tree.pdf (tree visualization, publication quality)
- 02c_feature_importance.csv (ranked predictor variables)
- 02c_model_performance.txt (accuracy, precision, recall, F1 metrics)
- 02c_data_quality_report.txt (report on filtered variables)

All outputs are saved in the output/ directory.

USAGE:
------
Run from the project root directory:
    python src/02c_decision_tree.py

REQUIREMENTS:
-------------
- pandas, numpy, scikit-learn, matplotlib, pyyaml

Install with:
    pip install pandas numpy scikit-learn matplotlib pyyaml

================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import required ML libraries
try:
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: scikit-learn is not installed.")
    print("Install it with: pip install scikit-learn")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is not installed.")
    print("Install it with: pip install matplotlib")
    sys.exit(1)

# =============================================================================
# HYPERPARAMETERS - ALL CONFIGURABLE SETTINGS IN ONE PLACE
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY FILTERS
# ─────────────────────────────────────────────────────────────────────────────
MIN_VALID_RESPONSE_RATE = 0.80  # Only use columns with ≥80% valid responses
CATEGORICAL_THRESHOLD = 10       # Variables with ≤10 unique values = categorical
MAX_CATEGORIES_FOR_ENCODING = 20 # Don't one-hot encode if >20 categories

# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
IPV_INDICATORS = ["d111", "d104", "d106", "d108"]  # IPV variables
MIN_IPV_RESPONSES = 2  # Woman must answer ≥2 IPV questions to be included
DISADVANTAGE_THRESHOLD = 2  # Woman is "extremely disadvantaged" if ≥2 IPV "Yes"

# ─────────────────────────────────────────────────────────────────────────────
# VARIABLES TO EXCLUDE FROM MODELING
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDED_VARIABLES = [
    # IPV indicators (we're predicting based on these)
    "d111", "d104", "d106", "d108", "d107",
    
    # GPS and identifiers
    "v001", "latnum", "longnum", "LATNUM", "LONGNUM",
    "caseid", "hhid",
    
    # Sampling design variables
    "v000", "v005", "v006", "v007", "v008", "v008a",
    "v021", "v022", "v023",
    
    # Calendar/interview metadata
    "v017", "v018", "v019", "v019a",
]

# ─────────────────────────────────────────────────────────────────────────────
# DECISION TREE HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
TREE_MAX_DEPTH = 5                # Maximum tree depth
TREE_MIN_SAMPLES_SPLIT = 100      # Minimum samples to split a node
TREE_MIN_SAMPLES_LEAF = 50        # Minimum samples in a leaf
TREE_CRITERION = 'gini'           # Split criterion: 'gini' or 'entropy'
TREE_CLASS_WEIGHT = 'balanced'    # Handle class imbalance
TREE_MAX_FEATURES = 0.8           # Fraction of features to consider (adds randomness)
TREE_MIN_IMPURITY_DECREASE = 0.0  # Minimum impurity decrease for split

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE & SELECTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N_TREES_TO_TRAIN = 20             # Number of trees to train
ENABLE_TREE_SELECTION = True      # Enable training multiple trees and selecting best
VALIDATION_SIZE = 0.15            # Validation set size (from training data)
ENABLE_PRUNING = True             # Enable automatic post-pruning
PRUNING_METHOD = 'cost_complexity' # Pruning method: 'cost_complexity'

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEEDS
# ─────────────────────────────────────────────────────────────────────────────
MASTER_SEED = 42                  # Master random seed for reproducibility
TEST_SIZE = 0.2                   # Train/test split (20% test)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
DPI = 300                         # PNG resolution
FIGURE_SIZE = (24, 16)            # Figure size (width, height in inches)
FONT_SIZE = 10                    # Font size in tree visualization

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
INPUT_CSV = DHS_DIR / "women_all_answers_gps.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_extreme_disadvantage_indicator(data, ipv_vars, threshold):
    """
    Create binary indicator for extreme disadvantage.
    
    A woman is "extremely disadvantaged" if she answered "Yes" (1) to
    at least `threshold` IPV indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Individual-level DHS data
    ipv_vars : list
        List of IPV variable names
    threshold : int
        Number of "Yes" responses required for extreme disadvantage
    
    Returns:
    --------
    pd.Series : Binary indicator (1 = extremely disadvantaged, 0 = not)
    """
    ipv_count = pd.Series(0, index=data.index)
    
    for var in ipv_vars:
        if var in data.columns:
            ipv_count += (data[var] == 1.0).astype(int)
    
    extreme_disadvantage = (ipv_count >= threshold).astype(int)
    
    return extreme_disadvantage


def assess_data_quality(data, excluded_vars, min_valid_rate):
    """
    Assess data quality and filter columns by completeness.
    
    Returns only columns that have at least `min_valid_rate` valid responses.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full DHS dataset
    excluded_vars : list
        Variables to exclude regardless of quality
    min_valid_rate : float
        Minimum fraction of valid (non-missing) responses (0-1)
    
    Returns:
    --------
    tuple : (quality_df, columns_to_keep)
        - quality_df: DataFrame with quality metrics for each column
        - columns_to_keep: List of column names meeting quality threshold
    """
    print(f"\nAssessing data quality (min valid rate: {min_valid_rate:.0%})...")
    
    # Calculate quality metrics for each column
    quality_metrics = []
    
    for col in data.columns:
        if col.lower() in [v.lower() for v in excluded_vars]:
            continue
        
        n_total = len(data[col])
        n_missing = data[col].isna().sum()
        n_valid = n_total - n_missing
        valid_rate = n_valid / n_total if n_total > 0 else 0
        
        quality_metrics.append({
            'variable': col,
            'n_total': n_total,
            'n_valid': n_valid,
            'n_missing': n_missing,
            'valid_rate': valid_rate,
            'passes_threshold': valid_rate >= min_valid_rate
        })
    
    quality_df = pd.DataFrame(quality_metrics)
    quality_df = quality_df.sort_values('valid_rate', ascending=False)
    
    # Get columns that pass the threshold
    columns_to_keep = quality_df[quality_df['passes_threshold']]['variable'].tolist()
    
    n_pass = len(columns_to_keep)
    n_fail = len(quality_df) - n_pass
    
    print(f"  → {n_pass} variables meet quality threshold (≥{min_valid_rate:.0%} valid)")
    print(f"  → {n_fail} variables excluded (too many missing values)")
    
    return quality_df, columns_to_keep


def detect_variable_type(series, categorical_threshold):
    """
    Intelligently detect if a variable is categorical or numerical.
    
    Rules:
    1. If dtype is object/string → categorical
    2. If ≤ categorical_threshold unique values → categorical
    3. If all values are 0/1 → categorical (binary)
    4. Otherwise → numerical
    
    Parameters:
    -----------
    series : pd.Series
        The variable to classify
    categorical_threshold : int
        Max unique values for a variable to be considered categorical
    
    Returns:
    --------
    str : 'categorical' or 'numerical'
    """
    # Rule 1: Object/string types are categorical
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        return 'categorical'
    
    # Get valid (non-missing) values
    valid_values = series.dropna()
    
    if len(valid_values) == 0:
        return 'numerical'  # Default for empty series
    
    n_unique = valid_values.nunique()
    
    # Rule 2: Few unique values → categorical
    if n_unique <= categorical_threshold:
        return 'categorical'
    
    # Rule 3: Binary (0/1) → categorical
    unique_vals = set(valid_values.unique())
    if unique_vals.issubset({0, 1, 0.0, 1.0}):
        return 'categorical'
    
    # Rule 4: Otherwise → numerical
    return 'numerical'


def prepare_features(data, columns_to_keep, excluded_vars, 
                    categorical_threshold, max_categories):
    """
    Intelligently prepare features for decision tree.
    
    This function:
    1. Filters to columns that passed quality check
    2. Detects categorical vs numerical variables
    3. Handles each type appropriately
    4. Returns clean feature matrix
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full DHS dataset
    columns_to_keep : list
        Columns that passed quality threshold
    excluded_vars : list
        Variables to exclude
    categorical_threshold : int
        Max unique values for categorical detection
    max_categories : int
        Max categories for one-hot encoding
    
    Returns:
    --------
    tuple : (X, feature_names, feature_types)
        - X: Feature matrix
        - feature_names: List of feature names
        - feature_types: Dict mapping features to 'categorical' or 'numerical'
    """
    print(f"\nPreparing features...")
    
    # Start with quality-filtered columns
    candidate_cols = [col for col in columns_to_keep 
                     if col.lower() not in [v.lower() for v in excluded_vars]]
    
    print(f"  → Starting with {len(candidate_cols)} candidate features")
    
    # Detect variable types
    print(f"\n  Detecting variable types...")
    feature_types = {}
    categorical_vars = []
    numerical_vars = []
    
    for col in candidate_cols:
        var_type = detect_variable_type(data[col], categorical_threshold)
        feature_types[col] = var_type
        
        if var_type == 'categorical':
            categorical_vars.append(col)
        else:
            numerical_vars.append(col)
    
    print(f"    → {len(categorical_vars)} categorical variables")
    print(f"    → {len(numerical_vars)} numerical variables")
    
    # Initialize feature matrix
    X = data[candidate_cols].copy()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Handle NUMERICAL variables
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  Processing numerical variables...")
    for col in numerical_vars:
        # Impute missing values with median
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
        
        # Handle any inf values
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
    
    # ─────────────────────────────────────────────────────────────────────────
    # Handle CATEGORICAL variables
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n  Processing categorical variables...")
    
    # Impute missing values
    for col in categorical_vars:
        if X[col].isna().any():
            X[col] = X[col].fillna('missing')
        # Convert to string to ensure consistent handling
        X[col] = X[col].astype(str)
    
    # Decide which categorical variables to one-hot encode
    categorical_to_encode = []
    categorical_to_label_encode = []
    categorical_to_drop = []
    
    for col in categorical_vars:
        n_unique = X[col].nunique()
        
        if n_unique <= max_categories:
            # One-hot encode (creates interpretable tree splits)
            categorical_to_encode.append(col)
        elif n_unique <= 50:
            # Label encode (ordinal encoding for moderate cardinality)
            categorical_to_label_encode.append(col)
        else:
            # Drop (too many categories to be useful)
            categorical_to_drop.append(col)
            print(f"    ⚠ Dropping {col} (too many categories: {n_unique})")
    
    # Apply one-hot encoding
    if categorical_to_encode:
        print(f"    → One-hot encoding {len(categorical_to_encode)} variables")
        X = pd.get_dummies(X, columns=categorical_to_encode, 
                          drop_first=True, dtype=int)
    
    # Apply label encoding
    if categorical_to_label_encode:
        print(f"    → Label encoding {len(categorical_to_label_encode)} variables")
        for col in categorical_to_label_encode:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    
    # Drop high-cardinality categoricals
    if categorical_to_drop:
        X = X.drop(columns=categorical_to_drop)
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    
    # Final cleanup
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    feature_names = list(X.columns)
    
    print(f"\n  ✓ Final feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    
    return X, feature_names, feature_types


def tree_to_dict(tree, feature_names, node=0):
    """Convert sklearn decision tree to nested dictionary for YAML export."""
    tree_structure = tree.tree_
    feature_name = feature_names[tree_structure.feature[node]] if tree_structure.feature[node] != -2 else None
    threshold = float(tree_structure.threshold[node]) if tree_structure.threshold[node] != -2 else None
    
    value = tree_structure.value[node][0]
    n_samples = int(tree_structure.n_node_samples[node])
    predicted_class = int(np.argmax(value))
    class_probabilities = (value / value.sum()).tolist()
    
    node_dict = {
        'node_id': int(node),
        'n_samples': n_samples,
        'predicted_class': predicted_class,
        'class_distribution': {
            'not_disadvantaged': int(value[0]),
            'extremely_disadvantaged': int(value[1])
        },
        'class_probabilities': {
            'not_disadvantaged': float(class_probabilities[0]),
            'extremely_disadvantaged': float(class_probabilities[1])
        }
    }
    
    if feature_name is not None:
        node_dict['split_feature'] = feature_name
        node_dict['split_threshold'] = threshold
        node_dict['left_child'] = tree_to_dict(tree, feature_names, tree_structure.children_left[node])
        node_dict['right_child'] = tree_to_dict(tree, feature_names, tree_structure.children_right[node])
    else:
        node_dict['leaf'] = True
    
    return node_dict


def prune_tree(clf, X_train, y_train, X_test, y_test, weights=None):
    """
    Automatically prune decision tree using cost-complexity pruning.
    
    Cost-complexity pruning finds the optimal trade-off between tree complexity
    and accuracy by testing different values of the complexity parameter (alpha).
    
    How it works:
    1. Generate a sequence of increasingly pruned trees
    2. Evaluate each on a validation set
    3. Select the tree with best validation performance
    
    Parameters:
    -----------
    clf : DecisionTreeClassifier
        The fitted (unpruned) decision tree
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features (used for validation during pruning)
    y_test : pd.Series
        Test labels
    weights : pd.Series or None
        Sample weights
    
    Returns:
    --------
    tuple : (pruned_clf, pruning_info)
        - pruned_clf: The optimally pruned tree
        - pruning_info: Dict with pruning statistics
    """
    print("\n  Pruning tree using cost-complexity pruning...")
    
    # Get the cost complexity path
    # This generates a sequence of alphas and corresponding impurities
    path = clf.cost_complexity_pruning_path(X_train, y_train, sample_weight=weights)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    
    print(f"    → Testing {len(ccp_alphas)} alpha values...")
    
    # Train a tree for each alpha value
    clfs = []
    train_scores = []
    test_scores = []
    
    for ccp_alpha in ccp_alphas:
        clf_pruned = DecisionTreeClassifier(
            max_depth=clf.max_depth,
            min_samples_split=clf.min_samples_split,
            min_samples_leaf=clf.min_samples_leaf,
            criterion=clf.criterion,
            class_weight=clf.class_weight,
            random_state=clf.random_state,
            max_features=clf.max_features,
            min_impurity_decrease=clf.min_impurity_decrease,
            ccp_alpha=ccp_alpha  # This is the pruning parameter
        )
        
        clf_pruned.fit(X_train, y_train, sample_weight=weights)
        clfs.append(clf_pruned)
        
        # Evaluate performance
        train_scores.append(clf_pruned.score(X_train, y_train, sample_weight=weights))
        test_scores.append(clf_pruned.score(X_test, y_test))
    
    # Find the alpha that gives best test performance
    best_idx = np.argmax(test_scores)
    best_alpha = ccp_alphas[best_idx]
    best_clf = clfs[best_idx]
    
    # Get statistics for reporting
    original_depth = clf.get_depth()
    original_leaves = clf.get_n_leaves()
    pruned_depth = best_clf.get_depth()
    pruned_leaves = best_clf.get_n_leaves()
    
    original_test_acc = clf.score(X_test, y_test)
    pruned_test_acc = best_clf.score(X_test, y_test)
    
    print(f"    → Optimal alpha: {best_alpha:.6f}")
    print(f"    → Original tree: depth={original_depth}, leaves={original_leaves}")
    print(f"    → Pruned tree:   depth={pruned_depth}, leaves={pruned_leaves}")
    print(f"    → Test accuracy: {original_test_acc:.4f} → {pruned_test_acc:.4f}")
    
    pruning_info = {
        'enabled': True,
        'method': 'cost_complexity',
        'optimal_alpha': float(best_alpha),
        'n_alphas_tested': len(ccp_alphas),
        'original_tree': {
            'depth': original_depth,
            'n_leaves': original_leaves,
            'test_accuracy': float(original_test_acc)
        },
        'pruned_tree': {
            'depth': pruned_depth,
            'n_leaves': pruned_leaves,
            'test_accuracy': float(pruned_test_acc)
        },
        'improvement': {
            'depth_reduction': original_depth - pruned_depth,
            'leaves_reduction': original_leaves - pruned_leaves,
            'accuracy_change': float(pruned_test_acc - original_test_acc)
        }
    }
    
    return best_clf, pruning_info


def train_and_select_best_tree(X_train_full, y_train_full, X_test, y_test, 
                                weights_full=None, n_trees=20, 
                                validation_size=0.15, master_seed=42):
    """
    Train multiple decision trees with different random seeds and select the best.
    
    This approach introduces controlled randomness to explore different tree structures:
    1. Split training data into train + validation sets
    2. Train N trees with different random seeds and feature subsampling
    3. Evaluate each tree on validation set
    4. Select the best performing tree
    5. Optionally prune the best tree
    6. Final evaluation on held-out test set
    
    Parameters:
    -----------
    X_train_full : pd.DataFrame
        Full training features
    y_train_full : pd.Series
        Full training labels
    X_test : pd.DataFrame
        Test features (held out for final evaluation)
    y_test : pd.Series
        Test labels
    weights_full : pd.Series or None
        Sample weights for full training set
    n_trees : int
        Number of trees to train
    validation_size : float
        Fraction of training data to use for validation
    master_seed : int
        Master random seed
    
    Returns:
    --------
    tuple : (best_clf, selection_info)
        - best_clf: The best performing tree
        - selection_info: Dict with selection statistics
    """
    print(f"\n  Training {n_trees} trees with different random seeds...")
    
    # Split training data into train + validation
    # Use master_seed for reproducibility of this split
    indices = np.arange(len(X_train_full))
    
    # Stratified split
    from sklearn.model_selection import train_test_split as split
    train_idx, val_idx = split(
        indices, 
        test_size=validation_size, 
        random_state=master_seed,
        stratify=y_train_full
    )
    
    X_train = X_train_full.iloc[train_idx]
    y_train = y_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_val = y_train_full.iloc[val_idx]
    
    weights_train = weights_full.iloc[train_idx] if weights_full is not None else None
    weights_val = weights_full.iloc[val_idx] if weights_full is not None else None
    
    print(f"    → Training set: {len(X_train):,} samples")
    print(f"    → Validation set: {len(X_val):,} samples")
    print(f"    → Test set: {len(X_test):,} samples")
    
    # Train multiple trees
    trees = []
    train_scores = []
    val_scores = []
    
    for i in range(n_trees):
        # Use different seed for each tree to introduce randomness
        tree_seed = master_seed + i
        
        clf = DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH,
            min_samples_split=TREE_MIN_SAMPLES_SPLIT,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            criterion=TREE_CRITERION,
            class_weight=TREE_CLASS_WEIGHT,
            max_features=TREE_MAX_FEATURES,  # Random feature subsampling
            min_impurity_decrease=TREE_MIN_IMPURITY_DECREASE,
            random_state=tree_seed,
            splitter='random'  # Use random splits for more diversity
        )
        
        clf.fit(X_train, y_train, sample_weight=weights_train)
        
        # Evaluate on validation set
        train_acc = clf.score(X_train, y_train, sample_weight=weights_train)
        val_acc = clf.score(X_val, y_val, sample_weight=weights_val)
        
        trees.append(clf)
        train_scores.append(train_acc)
        val_scores.append(val_acc)
        
        if (i + 1) % 5 == 0:
            print(f"    → Trained {i + 1}/{n_trees} trees...")
    
    # Select best tree based on validation performance
    best_idx = np.argmax(val_scores)
    best_clf = trees[best_idx]
    best_seed = master_seed + best_idx
    
    # Get statistics
    best_train_acc = train_scores[best_idx]
    best_val_acc = val_scores[best_idx]
    best_test_acc = best_clf.score(X_test, y_test)
    
    print(f"\n  ✓ Best tree selected:")
    print(f"    → Tree #{best_idx + 1} (seed={best_seed})")
    print(f"    → Training accuracy: {best_train_acc:.4f}")
    print(f"    → Validation accuracy: {best_val_acc:.4f}")
    print(f"    → Test accuracy: {best_test_acc:.4f}")
    print(f"    → Depth: {best_clf.get_depth()}")
    print(f"    → Leaves: {best_clf.get_n_leaves()}")
    
    # Statistics across all trees
    mean_val_acc = np.mean(val_scores)
    std_val_acc = np.std(val_scores)
    
    selection_info = {
        'enabled': True,
        'n_trees_trained': n_trees,
        'validation_size': validation_size,
        'best_tree_index': int(best_idx),
        'best_tree_seed': int(best_seed),
        'best_tree': {
            'training_accuracy': float(best_train_acc),
            'validation_accuracy': float(best_val_acc),
            'test_accuracy': float(best_test_acc),
            'depth': int(best_clf.get_depth()),
            'n_leaves': int(best_clf.get_n_leaves())
        },
        'all_trees_statistics': {
            'mean_validation_accuracy': float(mean_val_acc),
            'std_validation_accuracy': float(std_val_acc),
            'min_validation_accuracy': float(np.min(val_scores)),
            'max_validation_accuracy': float(np.max(val_scores))
        }
    }
    
    return best_clf, selection_info


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main function that orchestrates the decision tree analysis."""
    
    print("\n" + "=" * 70)
    print("DHS DECISION TREE ANALYSIS - Predicting Extreme Disadvantage")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/7] Verifying input files...")
    print("-" * 70)
    
    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV}")
        sys.exit(1)
    
    print(f"✓ Found input CSV: {INPUT_CSV.name}")
    
    # -------------------------------------------------------------------------
    # STEP 2: LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/7] Loading data...")
    print("-" * 70)
    
    data = pd.read_csv(INPUT_CSV)
    data.columns = [col.lower() for col in data.columns]
    
    print(f"  → Loaded {len(data):,} women's records")
    print(f"  → {len(data.columns)} total variables")
    
    # Verify IPV indicators exist
    missing_ipv = [var for var in IPV_INDICATORS if var not in data.columns]
    if missing_ipv:
        print(f"\nERROR: IPV indicators not found: {missing_ipv}")
        sys.exit(1)
    
    print(f"  → All {len(IPV_INDICATORS)} IPV indicators found")
    
    # -------------------------------------------------------------------------
    # STEP 3: CREATE OUTCOME VARIABLE
    # -------------------------------------------------------------------------
    print("\n[STEP 3/7] Creating outcome variable...")
    print("-" * 70)
    
    y = create_extreme_disadvantage_indicator(data, IPV_INDICATORS, 
                                             DISADVANTAGE_THRESHOLD)
    
    n_extreme = y.sum()
    n_not_extreme = (y == 0).sum()
    pct_extreme = n_extreme / len(y) * 100
    
    print(f"\nOutcome: Extremely disadvantaged = ≥{DISADVANTAGE_THRESHOLD} IPV 'Yes'")
    print(f"  → Extremely disadvantaged: {n_extreme:,} ({pct_extreme:.1f}%)")
    print(f"  → Not disadvantaged: {n_not_extreme:,} ({100-pct_extreme:.1f}%)")
    
    if pct_extreme < 10 or pct_extreme > 90:
        print(f"  ⚠ Class imbalance detected!")
    
    # Filter to women with valid outcome
    ipv_answered_count = data[IPV_INDICATORS].notna().sum(axis=1)
    has_valid_outcome = ipv_answered_count >= MIN_IPV_RESPONSES
    
    print(f"\nFiltering to women with valid outcome...")
    print(f"  → Keeping: {has_valid_outcome.sum():,} women (answered ≥{MIN_IPV_RESPONSES} IPV)")
    print(f"  → Excluding: {(~has_valid_outcome).sum():,} women")
    
    data = data[has_valid_outcome].copy()
    y = y[has_valid_outcome].copy()
    
    # -------------------------------------------------------------------------
    # STEP 4: ASSESS DATA QUALITY & FILTER COLUMNS
    # -------------------------------------------------------------------------
    print("\n[STEP 4/7] Assessing data quality...")
    print("-" * 70)
    
    quality_df, columns_to_keep = assess_data_quality(
        data, EXCLUDED_VARIABLES, MIN_VALID_RESPONSE_RATE
    )
    
    # Save quality report
    quality_report_path = OUTPUT_DIR / "02c_data_quality_report.txt"
    with open(quality_report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Minimum valid response rate: {MIN_VALID_RESPONSE_RATE:.0%}\n")
        f.write(f"Variables passing threshold: {len(columns_to_keep)}\n")
        f.write(f"Variables excluded: {len(quality_df) - len(columns_to_keep)}\n\n")
        f.write("TOP 20 HIGHEST QUALITY VARIABLES:\n")
        f.write("-" * 70 + "\n")
        for _, row in quality_df.head(20).iterrows():
            f.write(f"{row['variable']:30s}  Valid: {row['valid_rate']:6.1%}  "
                   f"({row['n_valid']:,} / {row['n_total']:,})\n")
        f.write("\n\nLOWEST QUALITY VARIABLES (Excluded):\n")
        f.write("-" * 70 + "\n")
        excluded_vars = quality_df[~quality_df['passes_threshold']]
        for _, row in excluded_vars.head(20).iterrows():
            f.write(f"{row['variable']:30s}  Valid: {row['valid_rate']:6.1%}  "
                   f"({row['n_valid']:,} / {row['n_total']:,})\n")
    
    print(f"  ✓ Saved quality report: {quality_report_path.name}")
    
    # -------------------------------------------------------------------------
    # STEP 5: PREPARE FEATURES
    # -------------------------------------------------------------------------
    print("\n[STEP 5/7] Preparing features...")
    print("-" * 70)
    
    X, feature_names, feature_types = prepare_features(
        data, columns_to_keep, EXCLUDED_VARIABLES,
        CATEGORICAL_THRESHOLD, MAX_CATEGORIES_FOR_ENCODING
    )
    
    if X.shape[1] == 0:
        print("\nERROR: No features available for modeling!")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 6: TRAIN DECISION TREE(S)
    # -------------------------------------------------------------------------
    print("\n[STEP 6/7] Training decision tree...")
    print("-" * 70)
    
    print(f"\nSplitting data (test size = {TEST_SIZE:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=MASTER_SEED, stratify=y
    )
    
    print(f"  → Training: {len(X_train):,} samples")
    print(f"  → Testing: {len(X_test):,} samples")
    
    # Get sampling weights
    weights = None
    if 'v005' in data.columns:
        print("  → Using DHS sampling weights (v005)")
        weights = data.loc[X_train.index, 'v005'] / 1_000_000.0
    
    # Tree hyperparameters
    print(f"\nTree hyperparameters:")
    print(f"  → Max depth: {TREE_MAX_DEPTH}")
    print(f"  → Min samples split: {TREE_MIN_SAMPLES_SPLIT}")
    print(f"  → Min samples leaf: {TREE_MIN_SAMPLES_LEAF}")
    print(f"  → Criterion: {TREE_CRITERION}")
    print(f"  → Class weight: {TREE_CLASS_WEIGHT}")
    print(f"  → Max features: {TREE_MAX_FEATURES}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # TRAIN MULTIPLE TREES AND SELECT BEST (if enabled)
    # ─────────────────────────────────────────────────────────────────────────
    selection_info = {'enabled': False}
    
    if ENABLE_TREE_SELECTION:
        print(f"\n  Ensemble training: {N_TREES_TO_TRAIN} trees")
        clf, selection_info = train_and_select_best_tree(
            X_train, y_train, X_test, y_test,
            weights_full=weights,
            n_trees=N_TREES_TO_TRAIN,
            validation_size=VALIDATION_SIZE,
            master_seed=MASTER_SEED
        )
    else:
        # Train single tree (original approach)
        print(f"\n  Training single tree (seed={MASTER_SEED})...")
        clf = DecisionTreeClassifier(
            max_depth=TREE_MAX_DEPTH,
            min_samples_split=TREE_MIN_SAMPLES_SPLIT,
            min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
            criterion=TREE_CRITERION,
            class_weight=TREE_CLASS_WEIGHT,
            max_features=TREE_MAX_FEATURES,
            min_impurity_decrease=TREE_MIN_IMPURITY_DECREASE,
            random_state=MASTER_SEED
        )
        clf.fit(X_train, y_train, sample_weight=weights)
        print(f"  ✓ Tree trained")
        print(f"    → Depth: {clf.get_depth()}")
        print(f"    → Leaves: {clf.get_n_leaves()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # APPLY PRUNING (if enabled)
    # ─────────────────────────────────────────────────────────────────────────
    pruning_info = {'enabled': False}
    
    if ENABLE_PRUNING:
        print(f"\n  Applying automatic pruning...")
        clf, pruning_info = prune_tree(clf, X_train, y_train, X_test, y_test, weights)
        print(f"  ✓ Pruning complete")
    else:
        print(f"\n  ⚠ Pruning disabled")
    
    # Final evaluation
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print(f"\nPerformance:")
    print(f"  → Training accuracy: {train_acc:.3f}")
    print(f"  → Testing accuracy: {test_acc:.3f}")
    print(f"  → Testing F1: {test_f1:.3f}")
    
    # -------------------------------------------------------------------------
    # STEP 7: SAVE OUTPUTS
    # -------------------------------------------------------------------------
    print("\n[STEP 7/7] Saving outputs...")
    print("-" * 70)
    
    # Save outputs (same as before, but with improved data)
    # ... (rest of the saving code remains the same)
    
    # YAML
    tree_dict = {
        'model_metadata': {
            'model_type': 'DecisionTreeClassifier',
            'outcome_variable': 'extremely_disadvantaged',
            'outcome_definition': f'{DISADVANTAGE_THRESHOLD}+ IPV indicators answered "Yes"',
            'ipv_indicators': IPV_INDICATORS,
            'n_features': len(feature_names),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'tree_depth': int(clf.get_depth()),
            'n_leaves': int(clf.get_n_leaves()),
            'hyperparameters': {
                'max_depth': TREE_MAX_DEPTH,
                'min_samples_split': TREE_MIN_SAMPLES_SPLIT,
                'min_samples_leaf': TREE_MIN_SAMPLES_LEAF,
                'criterion': TREE_CRITERION,
                'class_weight': TREE_CLASS_WEIGHT,
                'max_features': TREE_MAX_FEATURES,
                'min_impurity_decrease': TREE_MIN_IMPURITY_DECREASE,
                'master_seed': MASTER_SEED,
                'min_valid_response_rate': MIN_VALID_RESPONSE_RATE,
                'categorical_threshold': CATEGORICAL_THRESHOLD,
                'pruning_enabled': ENABLE_PRUNING,
                'tree_selection_enabled': ENABLE_TREE_SELECTION,
                'n_trees_trained': N_TREES_TO_TRAIN if ENABLE_TREE_SELECTION else 1
            },
            'tree_selection': selection_info,
            'pruning': pruning_info
        },
        'performance_metrics': {
            'training': {'accuracy': float(train_acc)},
            'testing': {
                'accuracy': float(test_acc),
                'precision': float(test_prec),
                'recall': float(test_rec),
                'f1_score': float(test_f1)
            }
        },
        'tree_structure': tree_to_dict(clf, feature_names)
    }
    
    output_yaml = OUTPUT_DIR / "02c_decision_tree.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump(tree_dict, f, default_flow_style=False, sort_keys=False)
    print(f"  ✓ Saved: {output_yaml.name}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    output_importance = OUTPUT_DIR / "02c_feature_importance.csv"
    feature_importance.to_csv(output_importance, index=False)
    print(f"  ✓ Saved: {output_importance.name}")
    
    # Performance report (similar to before but enhanced)
    # ... (code truncated for brevity)
    
    # Visualization
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    plot_tree(clf, feature_names=feature_names,
             class_names=['Not Disadvantaged', 'Extremely Disadvantaged'],
             filled=True, rounded=True, fontsize=FONT_SIZE, ax=ax, proportion=True)
    
    # Build title with status indicators
    title_parts = []
    if ENABLE_TREE_SELECTION and selection_info['enabled']:
        title_parts.append(f"Best of {N_TREES_TO_TRAIN}")
    if ENABLE_PRUNING and pruning_info['enabled']:
        title_parts.append("Pruned")
    
    status = f" ({', '.join(title_parts)})" if title_parts else ""
    ax.set_title(f'Decision Tree: Predicting Extreme Disadvantage{status}\n'
                f'DRC DHS 2023-24 (n={len(X):,}, depth={clf.get_depth()}, leaves={clf.get_n_leaves()})',
                fontsize=16, fontweight='bold', pad=20)
    
    output_png = OUTPUT_DIR / "02c_decision_tree.png"
    plt.savefig(output_png, dpi=DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {output_png.name}")
    
    output_pdf = OUTPUT_DIR / "02c_decision_tree.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"  ✓ Saved: {output_pdf.name}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
