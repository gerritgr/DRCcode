#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Decision Tree Analysis - Predicting Extreme Disadvantage
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script builds a decision tree classifier to predict "extremely disadvantaged"
women in the DRC 2023-24 DHS survey. The analysis:

1. DEFINES EXTREME DISADVANTAGE:
   - Women who answered "Yes" to 2 or more IPV indicators (D111, D104, D106, D108)
   - Creates a binary outcome: 1 = extremely disadvantaged, 0 = not

2. BUILDS A DECISION TREE:
   - Uses all other DHS variables as potential predictors
   - Excludes IPV indicators and user-specified variables
   - Applies DHS sampling weights for nationally representative results

3. OUTPUTS:
   - Tree structure as YAML (2c_decision_tree.yaml)
   - Tree visualization as PNG and PDF (2c_decision_tree.png/pdf)
   - Feature importance rankings
   - Model performance metrics

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (created by convert.py)

OUTPUTS:
--------
- 2c_decision_tree.yaml (tree structure in YAML format)
- 2c_decision_tree.png (tree visualization, high resolution)
- 2c_decision_tree.pdf (tree visualization, publication quality)
- 2c_feature_importance.csv (ranked predictor variables)
- 2c_model_performance.txt (accuracy, precision, recall, F1 metrics)

All outputs are saved in the output/ directory.

USAGE:
------
Run from the project root directory:
    python src/2c_decision_tree.py

Or from the src directory:
    python 2c_decision_tree.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (machine learning)
- matplotlib (plotting)
- pyyaml (YAML export)

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
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, confusion_matrix, 
        accuracy_score, precision_score, recall_score, f1_score
    )
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
# CONFIGURATION
# =============================================================================

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
INPUT_CSV = DHS_DIR / "women_all_answers_gps.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IPV INDICATORS that define extreme disadvantage
# Women who answer "Yes" to 2+ of these are classified as extremely disadvantaged
IPV_INDICATORS = ["d111", "d104", "d106", "d108"]

# EXCLUDED VARIABLES (cannot be used as predictors)
# These are excluded from the decision tree analysis
EXCLUDED_VARIABLES = [
    # IPV indicators (we're predicting based on these, so exclude them)
    "d111", "d104", "d106", "d108",
    
    # Other violence-related variables that would leak information
    "d107",  # Severe violence indicator
    
    # GPS and cluster identifiers (not meaningful predictors)
    "v001",      # Cluster number
    "latnum",    # Latitude
    "longnum",   # Longitude
    "LATNUM",    # Latitude (uppercase)
    "LONGNUM",   # Longitude (uppercase)
    
    # Case identifiers
    "caseid",    # Case ID
    "hhid",      # Household ID
    
    # Sampling design variables (technical, not substantive predictors)
    "v000",      # Country code
    "v005",      # Women's individual sample weight
    "v006",      # Month of interview
    "v007",      # Year of interview
    "v008",      # Date of interview (CMC)
    "v021",      # Primary sampling unit
    "v022",      # Sample strata
    "v023",      # Sample domain
    
    # Add any other variables you want to exclude below:
    # Example: "v012",  # Respondent's age (if you want to exclude it)
]

# DECISION TREE PARAMETERS
TREE_MAX_DEPTH = 5              # Maximum depth of tree (prevents overfitting)
TREE_MIN_SAMPLES_SPLIT = 100    # Minimum samples required to split a node
TREE_MIN_SAMPLES_LEAF = 50      # Minimum samples required in a leaf node
RANDOM_STATE = 42               # For reproducibility
TEST_SIZE = 0.2                 # Fraction of data for testing (20%)

# VISUALIZATION PARAMETERS
DPI = 300                       # Resolution for PNG output
FIGURE_SIZE = (24, 16)          # Size of tree visualization (width, height in inches)
FONT_SIZE = 10                  # Font size for tree nodes

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_extreme_disadvantage_indicator(data, ipv_vars):
    """
    Create binary indicator for extreme disadvantage.
    
    A woman is "extremely disadvantaged" if she answered "Yes" (1) to
    2 or more IPV indicators.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Individual-level DHS data
    ipv_vars : list
        List of IPV variable names (e.g., ['d111', 'd104', 'd106', 'd108'])
    
    Returns:
    --------
    pd.Series : Binary indicator (1 = extremely disadvantaged, 0 = not)
    """
    # Count how many IPV indicators each woman answered "Yes" to
    # Only count valid responses (0 or 1, not NaN)
    ipv_count = pd.Series(0, index=data.index)
    
    for var in ipv_vars:
        if var in data.columns:
            # Add 1 for each "Yes" response
            ipv_count += (data[var] == 1.0).astype(int)
    
    # Extremely disadvantaged = 2 or more "Yes" responses
    extreme_disadvantage = (ipv_count >= 2).astype(int)
    
    return extreme_disadvantage


def prepare_features(data, excluded_vars):
    """
    Prepare feature matrix for decision tree.
    
    This function:
    1. Removes excluded variables
    2. Handles missing values
    3. Encodes categorical variables
    4. Returns clean feature matrix
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full DHS dataset
    excluded_vars : list
        Variables to exclude from analysis
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix (cleaned and encoded)
    feature_names : list
        Names of features (for interpretation)
    """
    # Start with all columns
    candidate_features = [col for col in data.columns 
                         if col.lower() not in [v.lower() for v in excluded_vars]]
    
    print(f"\n  Candidate features: {len(candidate_features)} variables")
    
    # Extract feature matrix
    X = data[candidate_features].copy()
    
    # Handle missing values
    print("\n  Handling missing values...")
    
    # For numeric columns: impute with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
    
    # For categorical/object columns: impute with mode or create 'missing' category
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna('missing')
    
    # Convert categorical variables to numeric (one-hot encoding for low cardinality)
    print("\n  Encoding categorical variables...")
    categorical_to_encode = []
    for col in categorical_cols:
        n_unique = X[col].nunique()
        if n_unique <= 10:  # Only one-hot encode if <= 10 categories
            categorical_to_encode.append(col)
        else:
            # For high cardinality, drop or use label encoding
            print(f"    ⚠ Dropping {col} (too many categories: {n_unique})")
            X = X.drop(columns=[col])
    
    if categorical_to_encode:
        X = pd.get_dummies(X, columns=categorical_to_encode, drop_first=True)
        print(f"    → One-hot encoded {len(categorical_to_encode)} categorical variables")
    
    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"\n  ✓ Final feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    
    return X, list(X.columns)


def tree_to_dict(tree, feature_names, node=0):
    """
    Convert sklearn decision tree to nested dictionary (for YAML export).
    
    Parameters:
    -----------
    tree : sklearn.tree.DecisionTreeClassifier
        Fitted decision tree
    feature_names : list
        Names of features
    node : int
        Current node index (for recursion)
    
    Returns:
    --------
    dict : Tree structure as nested dictionary
    """
    tree_structure = tree.tree_
    feature_name = feature_names[tree_structure.feature[node]] if tree_structure.feature[node] != -2 else None
    threshold = float(tree_structure.threshold[node]) if tree_structure.threshold[node] != -2 else None
    
    # Get class distribution at this node
    value = tree_structure.value[node][0]
    n_samples = int(tree_structure.n_node_samples[node])
    
    # Class prediction (majority class)
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
    
    # If not a leaf node, add split information and recurse
    if feature_name is not None:
        node_dict['split_feature'] = feature_name
        node_dict['split_threshold'] = threshold
        node_dict['left_child'] = tree_to_dict(tree, feature_names, tree_structure.children_left[node])
        node_dict['right_child'] = tree_to_dict(tree, feature_names, tree_structure.children_right[node])
    else:
        node_dict['leaf'] = True
    
    return node_dict


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the decision tree analysis.
    """
    
    print("\n" + "=" * 70)
    print("DHS DECISION TREE ANALYSIS - Predicting Extreme Disadvantage")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/6] Verifying input files...")
    print("-" * 70)
    
    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV}")
        print("\nPlease run DATA/DHS/convert.py first to generate this file.")
        sys.exit(1)
    
    print(f"✓ Found input CSV: {INPUT_CSV.name}")
    print(f"  Location: {INPUT_CSV}")
    
    # -------------------------------------------------------------------------
    # STEP 2: LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/6] Loading data...")
    print("-" * 70)
    
    print(f"\nReading: {INPUT_CSV.name}")
    data = pd.read_csv(INPUT_CSV)
    print(f"  → Loaded {len(data):,} women's records")
    print(f"  → {len(data.columns)} total variables")
    
    # Convert column names to lowercase for consistency
    data.columns = [col.lower() for col in data.columns]
    
    # Verify IPV indicators exist
    missing_ipv = [var for var in IPV_INDICATORS if var not in data.columns]
    if missing_ipv:
        print(f"\nERROR: IPV indicators not found: {missing_ipv}")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)
    
    print(f"\n✓ All {len(IPV_INDICATORS)} IPV indicators found")
    
    # -------------------------------------------------------------------------
    # STEP 3: CREATE OUTCOME VARIABLE
    # -------------------------------------------------------------------------
    print("\n[STEP 3/6] Creating outcome variable (extreme disadvantage)...")
    print("-" * 70)
    
    # Create the extreme disadvantage indicator
    y = create_extreme_disadvantage_indicator(data, IPV_INDICATORS)
    
    # Report statistics
    n_extreme = y.sum()
    n_not_extreme = (y == 0).sum()
    pct_extreme = n_extreme / len(y) * 100
    
    print(f"\nOutcome variable created:")
    print(f"  → Extremely disadvantaged (2+ IPV 'Yes'): {n_extreme:,} women ({pct_extreme:.1f}%)")
    print(f"  → Not extremely disadvantaged: {n_not_extreme:,} women ({100-pct_extreme:.1f}%)")
    
    # Check for class imbalance
    if pct_extreme < 10 or pct_extreme > 90:
        print(f"\n  ⚠ WARNING: Class imbalance detected!")
        print(f"    Model may have difficulty with minority class")
    
    # Filter to women who have valid outcome (answered at least 2 IPV questions)
    # Count how many IPV questions each woman answered
    ipv_answered_count = data[IPV_INDICATORS].notna().sum(axis=1)
    has_valid_outcome = ipv_answered_count >= 2
    
    print(f"\nFiltering to women with valid outcome...")
    print(f"  → {has_valid_outcome.sum():,} women answered 2+ IPV questions")
    print(f"  → {(~has_valid_outcome).sum():,} women excluded (answered <2 IPV questions)")
    
    data = data[has_valid_outcome].copy()
    y = y[has_valid_outcome].copy()
    
    # -------------------------------------------------------------------------
    # STEP 4: PREPARE FEATURES
    # -------------------------------------------------------------------------
    print("\n[STEP 4/6] Preparing feature matrix...")
    print("-" * 70)
    
    print(f"\nExcluding {len(EXCLUDED_VARIABLES)} variables:")
    for var in EXCLUDED_VARIABLES[:10]:  # Show first 10
        print(f"  • {var}")
    if len(EXCLUDED_VARIABLES) > 10:
        print(f"  ... and {len(EXCLUDED_VARIABLES) - 10} more")
    
    X, feature_names = prepare_features(data, EXCLUDED_VARIABLES)
    
    # Check that we have features
    if X.shape[1] == 0:
        print("\nERROR: No features available for modeling!")
        print("Check your EXCLUDED_VARIABLES list.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 5: TRAIN DECISION TREE
    # -------------------------------------------------------------------------
    print("\n[STEP 5/6] Training decision tree...")
    print("-" * 70)
    
    # Split data into training and testing sets
    print(f"\nSplitting data (test size = {TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  → Training set: {len(X_train):,} samples")
    print(f"  → Testing set: {len(X_test):,} samples")
    
    # Get sampling weights if available
    if 'v005' in data.columns:
        print("\n✓ Using DHS sampling weights (v005)")
        weights = data.loc[X_train.index, 'v005'] / 1_000_000.0
        weights_test = data.loc[X_test.index, 'v005'] / 1_000_000.0
    else:
        print("\n⚠ No sampling weights found, using equal weights")
        weights = None
        weights_test = None
    
    # Train decision tree
    print(f"\nTraining decision tree...")
    print(f"  → Max depth: {TREE_MAX_DEPTH}")
    print(f"  → Min samples to split: {TREE_MIN_SAMPLES_SPLIT}")
    print(f"  → Min samples per leaf: {TREE_MIN_SAMPLES_LEAF}")
    
    clf = DecisionTreeClassifier(
        max_depth=TREE_MAX_DEPTH,
        min_samples_split=TREE_MIN_SAMPLES_SPLIT,
        min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train, sample_weight=weights)
    
    print(f"  ✓ Tree trained successfully")
    print(f"  → Tree depth: {clf.get_depth()}")
    print(f"  → Number of leaves: {clf.get_n_leaves()}")
    
    # Make predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Evaluate on training set
    print(f"\nTraining set performance:")
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"  → Accuracy: {train_acc:.3f}")
    
    # Evaluate on testing set
    print(f"\nTesting set performance:")
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test)
    test_rec = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print(f"  → Accuracy: {test_acc:.3f}")
    print(f"  → Precision: {test_prec:.3f}")
    print(f"  → Recall: {test_rec:.3f}")
    print(f"  → F1-Score: {test_f1:.3f}")
    
    # -------------------------------------------------------------------------
    # STEP 6: SAVE OUTPUTS
    # -------------------------------------------------------------------------
    print("\n[STEP 6/6] Saving outputs...")
    print("-" * 70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 1: Tree structure as YAML
    # ─────────────────────────────────────────────────────────────────────────
    print("\n1. Saving tree structure (YAML)...")
    
    tree_dict = {
        'model_metadata': {
            'model_type': 'DecisionTreeClassifier',
            'outcome_variable': 'extremely_disadvantaged',
            'outcome_definition': '2 or more IPV indicators answered "Yes"',
            'ipv_indicators': IPV_INDICATORS,
            'n_features': len(feature_names),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'tree_depth': int(clf.get_depth()),
            'n_leaves': int(clf.get_n_leaves()),
            'parameters': {
                'max_depth': TREE_MAX_DEPTH,
                'min_samples_split': TREE_MIN_SAMPLES_SPLIT,
                'min_samples_leaf': TREE_MIN_SAMPLES_LEAF,
                'random_state': RANDOM_STATE
            }
        },
        'performance_metrics': {
            'training': {
                'accuracy': float(train_acc)
            },
            'testing': {
                'accuracy': float(test_acc),
                'precision': float(test_prec),
                'recall': float(test_rec),
                'f1_score': float(test_f1)
            }
        },
        'tree_structure': tree_to_dict(clf, feature_names)
    }
    
    output_yaml = OUTPUT_DIR / "2c_decision_tree.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump(tree_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"  ✓ Saved: {output_yaml.name}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 2: Feature importance
    # ─────────────────────────────────────────────────────────────────────────
    print("\n2. Saving feature importance...")
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    output_importance = OUTPUT_DIR / "2c_feature_importance.csv"
    feature_importance.to_csv(output_importance, index=False)
    
    print(f"  ✓ Saved: {output_importance.name}")
    print(f"\n  Top 10 most important features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:40s} {row['importance']:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 3: Model performance report
    # ─────────────────────────────────────────────────────────────────────────
    print("\n3. Saving model performance report...")
    
    output_performance = OUTPUT_DIR / "2c_model_performance.txt"
    with open(output_performance, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DECISION TREE MODEL PERFORMANCE\n")
        f.write("Predicting Extreme Disadvantage in DRC DHS 2023-24\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OUTCOME DEFINITION:\n")
        f.write("-" * 70 + "\n")
        f.write("Extremely disadvantaged = Women who answered 'Yes' to 2+ IPV indicators\n")
        f.write(f"IPV indicators: {', '.join(IPV_INDICATORS)}\n\n")
        
        f.write("DATASET:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total samples: {len(X):,}\n")
        f.write(f"  - Training: {len(X_train):,} ({(1-TEST_SIZE)*100:.0f}%)\n")
        f.write(f"  - Testing: {len(X_test):,} ({TEST_SIZE*100:.0f}%)\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write(f"Class distribution:\n")
        f.write(f"  - Not disadvantaged: {n_not_extreme:,} ({100-pct_extreme:.1f}%)\n")
        f.write(f"  - Extremely disadvantaged: {n_extreme:,} ({pct_extreme:.1f}%)\n\n")
        
        f.write("TREE PARAMETERS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Max depth: {TREE_MAX_DEPTH}\n")
        f.write(f"Min samples to split: {TREE_MIN_SAMPLES_SPLIT}\n")
        f.write(f"Min samples per leaf: {TREE_MIN_SAMPLES_LEAF}\n")
        f.write(f"Actual tree depth: {clf.get_depth()}\n")
        f.write(f"Number of leaves: {clf.get_n_leaves()}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 70 + "\n")
        f.write("Training Set:\n")
        f.write(f"  Accuracy: {train_acc:.4f}\n\n")
        
        f.write("Testing Set:\n")
        f.write(f"  Accuracy:  {test_acc:.4f}\n")
        f.write(f"  Precision: {test_prec:.4f}\n")
        f.write(f"  Recall:    {test_rec:.4f}\n")
        f.write(f"  F1-Score:  {test_f1:.4f}\n\n")
        
        f.write("CONFUSION MATRIX (Testing Set):\n")
        f.write("-" * 70 + "\n")
        cm = confusion_matrix(y_test, y_pred_test)
        f.write(f"                    Predicted: No   Predicted: Yes\n")
        f.write(f"Actual: No          {cm[0,0]:8d}      {cm[0,1]:8d}\n")
        f.write(f"Actual: Yes         {cm[1,0]:8d}      {cm[1,1]:8d}\n\n")
        
        f.write("CLASSIFICATION REPORT (Testing Set):\n")
        f.write("-" * 70 + "\n")
        f.write(classification_report(
            y_test, y_pred_test, 
            target_names=['Not Disadvantaged', 'Extremely Disadvantaged']
        ))
        f.write("\n")
        
        f.write("TOP 20 MOST IMPORTANT FEATURES:\n")
        f.write("-" * 70 + "\n")
        for idx, row in feature_importance.head(20).iterrows():
            f.write(f"{row['feature']:50s} {row['importance']:.6f}\n")
    
    print(f"  ✓ Saved: {output_performance.name}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT 4: Tree visualization (PNG and PDF)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n4. Creating tree visualizations...")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=['Not Disadvantaged', 'Extremely Disadvantaged'],
        filled=True,
        rounded=True,
        fontsize=FONT_SIZE,
        ax=ax,
        proportion=True  # Show proportions instead of counts
    )
    
    ax.set_title(
        'Decision Tree: Predicting Extreme Disadvantage in Women\n'
        f'DRC DHS 2023-24 (n={len(X):,}, depth={clf.get_depth()})',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Save as PNG
    output_png = OUTPUT_DIR / "2c_decision_tree.png"
    plt.savefig(output_png, dpi=DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {output_png.name}")
    
    # Save as PDF
    output_pdf = OUTPUT_DIR / "2c_decision_tree.pdf"
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"  ✓ Saved: {output_pdf.name}")
    
    plt.close()
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DECISION TREE ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    print(f"  1. 2c_decision_tree.yaml")
    print(f"     → Tree structure in YAML format")
    print(f"     → {clf.get_depth()} levels, {clf.get_n_leaves()} leaf nodes")
    
    print(f"\n  2. 2c_decision_tree.png")
    print(f"     → Tree visualization (high resolution)")
    
    print(f"\n  3. 2c_decision_tree.pdf")
    print(f"     → Tree visualization (publication quality)")
    
    print(f"\n  4. 2c_feature_importance.csv")
    print(f"     → Ranked list of {len(feature_names)} predictor variables")
    
    print(f"\n  5. 2c_model_performance.txt")
    print(f"     → Detailed performance metrics and statistics")
    
    print(f"\n📈 MODEL SUMMARY:")
    print(f"  → Outcome: Extremely disadvantaged (2+ IPV 'Yes')")
    print(f"  → Prevalence: {pct_extreme:.1f}% ({n_extreme:,}/{len(y):,} women)")
    print(f"  → Testing accuracy: {test_acc:.3f}")
    print(f"  → Testing F1-score: {test_f1:.3f}")
    
    print("\n✅ Analysis complete!\n")


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)