#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Summary Statistics Generator - DRC 2023-24 Survey
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script creates comprehensive summary statistics visualizations for ALL
variables in the DHS dataset. For each variable, it automatically determines
the best visualization approach:

1. BINARY/CATEGORICAL VARIABLES (≤10 unique values):
   - Creates bar charts showing frequency distribution
   - Displays "No response", "Yes", "No", and other categories
   - Shows both counts and percentages

2. CONTINUOUS/MULTI-VALUE VARIABLES (>10 unique values):
   - Creates histograms with distribution shape
   - Shows descriptive statistics (mean, median, std, min, max)
   - Includes quartile information

3. SPECIAL HANDLING:
   - Missing values are explicitly shown as "No response"
   - Variables with all missing data are flagged
   - GPS coordinates and identifiers get appropriate formatting

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (created by convert.py)

OUTPUTS:
--------
- 2d_summary_statistics.txt (comprehensive text file with all variable summaries)
  Easy-to-read format with statistics for each variable

All outputs are saved in the output/ directory.

USAGE:
------
Run from the project root directory:
    python src/2d_summary_statistics.py

Or from the src directory:
    python 2d_summary_statistics.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- numpy (numerical operations)

Install with:
    pip install pandas numpy

================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
VARIABLE_NAMES_MAP = DHS_DIR / "variable_names_map.txt"  # Optional: descriptions of variables

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Text formatting parameters
LINE_WIDTH = 80                 # Width of separator lines
SECTION_CHAR = '='              # Character for major section separators
SUBSECTION_CHAR = '-'           # Character for minor section separators
CATEGORICAL_THRESHOLD = 15      # Variables with ≤10 unique values are categorical

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_variable_descriptions(filepath):
    """
    Load variable descriptions from the DHS variable names mapping file.
    
    The mapping file contains variable names and their descriptions in a format like:
    V000                   Country code and phase
    V001                   Cluster number
    V002                   Household number
    ...
    
    Some variables may have value labels on subsequent indented lines like:
                               1  15-19
                               2  20-24
    
    This function extracts ONLY the variable label (main description),
    not the value labels (answer options) or metadata columns.
    
    Parameters:
    -----------
    filepath : Path
        Path to the variable_names_map.txt file
    
    Returns:
    --------
    dict : Dictionary mapping variable names (lowercase) to descriptions
           Returns empty dict if file doesn't exist or can't be parsed
    """
    import re  # For pattern matching and cleaning
    
    if not filepath.exists():
        print(f"  ⚠ Variable descriptions file not found: {filepath.name}")
        print(f"    Proceeding without variable descriptions...")
        return {}
    
    print(f"  Loading variable descriptions from: {filepath.name}")
    
    var_descriptions = {}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n\r')
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Skip lines that start with significant indentation
                # These are typically value labels (answer options)
                # Variable definitions start at the beginning of the line
                if len(line) > 0 and line[0] in (' ', '\t'):
                    # This is an indented line (likely a value label)
                    continue
                
                # Skip separator lines or header lines
                if line.strip().startswith('-') or line.strip().startswith('='):
                    continue
                
                # Skip lines that are too short to be variable definitions
                if len(line.strip()) < 10:
                    continue
                
                # Try to parse variable name and description
                # Format is typically: "VARNAME" followed by spaces, then description
                # Example: "V000                   Country code and phase"
                
                # Split on whitespace, get first token as potential variable name
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                
                var_name = parts[0].strip()
                description = parts[1].strip()
                
                # Clean the description by removing metadata columns
                # DHS codebook format often has: "Description    Start Len Type ..."
                # We want only the description part, not the technical metadata
                
                # Common patterns to remove (these are column values in the codebook):
                # - Multiple spaces followed by numbers (position/length info)
                # - "AN" or "N" (data type indicators)
                # - "Yes/No" patterns at the end
                
                # Strategy: Keep only the text before we hit obvious metadata
                # Metadata typically starts with multiple spaces followed by digits
                
                # Find where metadata starts (2+ spaces followed by digit)
                metadata_pattern = r'\s{2,}\d+'
                match = re.search(metadata_pattern, description)
                
                if match:
                    # Cut off everything from the metadata onwards
                    description = description[:match.start()].strip()
                
                # Additional cleanup: remove trailing "Yes/No" if present
                description = re.sub(r'\s+(Yes|No)\s*$', '', description).strip()
                
                # Additional validation: variable names should be alphanumeric
                # and typically uppercase (though we'll store as lowercase)
                if not var_name:
                    continue
                if not any(c.isalpha() for c in var_name):
                    # Skip if no letters (probably not a variable name)
                    continue
                
                # Skip common header/footer text patterns
                skip_patterns = [
                    'level', 'record', 'item name', 'item label',
                    'start', 'len', 'type', 'occ', 'dec', 'char', 'fill',
                    'data item', 'case identification', 'last modified'
                ]
                if any(pattern in var_name.lower() for pattern in skip_patterns):
                    continue
                if any(pattern in description.lower()[:50] for pattern in ['last modified', 'level label']):
                    continue
                
                # Store both uppercase and lowercase versions
                # DHS variables are typically uppercase, but our data may be lowercase
                var_descriptions[var_name.lower()] = description
                
        print(f"    → Loaded descriptions for {len(var_descriptions):,} variables")
        
    except Exception as e:
        print(f"  ⚠ Error reading variable descriptions: {e}")
        print(f"    Proceeding without variable descriptions...")
        return {}
    
    return var_descriptions

def identify_variable_type(series):
    """
    Identify what type of variable we're dealing with.
    
    This function examines a pandas Series and determines whether it's:
    - Binary (Yes/No question)
    - Categorical (few distinct values, like education level)
    - Continuous (many values, like age or income)
    - Identifier (like case IDs - these should be skipped)
    
    Parameters:
    -----------
    series : pd.Series
        A column from the DHS dataset
    
    Returns:
    --------
    dict : Dictionary with 'type' and metadata about the variable
    """
    # Count unique values (excluding NaN)
    n_unique = series.nunique(dropna=True)
    n_total = len(series)
    n_missing = series.isna().sum()
    n_valid = n_total - n_missing
    
    # Get the data type
    dtype = series.dtype
    
    # Check if it's an identifier (many unique values that are IDs)
    identifier_keywords = ['id', 'caseid', 'hhid', 'cluster']
    is_identifier = any(keyword in series.name.lower() for keyword in identifier_keywords)
    
    if is_identifier:
        return {
            'type': 'identifier',
            'n_unique': n_unique,
            'n_missing': n_missing,
            'n_valid': n_valid
        }
    
    # Check if all values are missing
    if n_valid == 0:
        return {
            'type': 'all_missing',
            'n_unique': 0,
            'n_missing': n_missing,
            'n_valid': 0
        }
    
    # Check if it's binary (0/1 or Yes/No)
    # Common for DHS violence indicators
    if n_unique <= 2:
        unique_vals = series.dropna().unique()
        is_binary = all(v in [0, 1, 0.0, 1.0, '0', '1', 'yes', 'no', 'Yes', 'No'] 
                       for v in unique_vals)
        
        if is_binary:
            return {
                'type': 'binary',
                'n_unique': n_unique,
                'n_missing': n_missing,
                'n_valid': n_valid,
                'unique_values': list(unique_vals)
            }
    
    # Check if it's categorical (≤10 unique values)
    # Examples: education level (none, primary, secondary, higher)
    #           region (province names)
    if n_unique <= CATEGORICAL_THRESHOLD:
        return {
            'type': 'categorical',
            'n_unique': n_unique,
            'n_missing': n_missing,
            'n_valid': n_valid
        }
    
    # Check if it's numeric continuous (>10 unique values and numeric type)
    # Examples: age, weight, income, number of children
    if pd.api.types.is_numeric_dtype(dtype) and n_unique > CATEGORICAL_THRESHOLD:
        return {
            'type': 'continuous',
            'n_unique': n_unique,
            'n_missing': n_missing,
            'n_valid': n_valid,
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75)
        }
    
    # Otherwise, treat as categorical with many levels
    # Example: open-ended text responses (rare in DHS)
    return {
        'type': 'categorical_many',
        'n_unique': n_unique,
        'n_missing': n_missing,
        'n_valid': n_valid
    }


def format_binary_variable(series, var_name, var_description=None):
    """
    Format summary statistics for binary variables (Yes/No questions) as text.
    
    Binary variables are the most common in DHS violence indicators.
    We want to clearly show:
    - How many women answered "Yes"
    - How many women answered "No"
    - How many women didn't answer (missing data)
    
    Parameters:
    -----------
    series : pd.Series
        The variable data (should be 0/1 or Yes/No)
    var_name : str
        Name of the variable (for the header)
    var_description : str, optional
        Human-readable description of the variable
    
    Returns:
    --------
    str : Formatted text summary
    """
    # Count the responses
    # Convert to numeric first to handle any string values
    series_numeric = pd.to_numeric(series, errors='coerce')
    
    n_yes = (series_numeric == 1).sum()
    n_no = (series_numeric == 0).sum()
    n_missing = series_numeric.isna().sum()
    n_total = len(series)
    
    # Calculate percentages
    pct_yes = n_yes / n_total * 100 if n_total > 0 else 0
    pct_no = n_no / n_total * 100 if n_total > 0 else 0
    pct_missing = n_missing / n_total * 100 if n_total > 0 else 0
    
    # Build the text output
    output = []
    output.append(f"VARIABLE: {var_name}")
    if var_description:
        output.append(f"DESCRIPTION: {var_description}")
    output.append(f"TYPE: Binary (Yes/No)")
    output.append(f"TOTAL RESPONSES: {n_total:,}")
    output.append("")
    output.append("DISTRIBUTION:")
    output.append(f"  No response:  {n_missing:8,}  ({pct_missing:6.2f}%)")
    output.append(f"  Yes:          {n_yes:8,}  ({pct_yes:6.2f}%)")
    output.append(f"  No:           {n_no:8,}  ({pct_no:6.2f}%)")
    
    return "\n".join(output)


def format_categorical_variable(series, var_name, var_description=None):
    """
    Format summary statistics for categorical variables as text.
    
    Categorical variables have a fixed set of options, like:
    - Education level: None, Primary, Secondary, Higher
    - Wealth quintile: Poorest, Poorer, Middle, Richer, Richest
    - Region: Different provinces
    
    We show the frequency of each category.
    
    Parameters:
    -----------
    series : pd.Series
        The variable data
    var_name : str
        Name of the variable (for the header)
    var_description : str, optional
        Human-readable description of the variable
    
    Returns:
    --------
    str : Formatted text summary
    """
    # Get value counts (excluding NaN initially)
    value_counts = series.value_counts(dropna=True).sort_values(ascending=False)
    
    # Count missing values
    n_missing = series.isna().sum()
    n_total = len(series)
    
    # Build the text output
    output = []
    output.append(f"VARIABLE: {var_name}")
    if var_description:
        output.append(f"DESCRIPTION: {var_description}")
    output.append(f"TYPE: Categorical ({len(value_counts)} unique values)")
    output.append(f"TOTAL RESPONSES: {n_total:,}")
    output.append("")
    output.append("DISTRIBUTION:")
    
    # Show missing values first if any
    if n_missing > 0:
        pct_missing = n_missing / n_total * 100
        output.append(f"  No response:  {n_missing:8,}  ({pct_missing:6.2f}%)")
    
    # Show top categories (limit to 20 for readability)
    max_display = 20
    for i, (category, count) in enumerate(value_counts.items()):
        if i >= max_display:
            # Show "Other" for remaining categories
            # Use .iloc[] for positional slicing (not label-based slicing)
            other_count = value_counts.iloc[max_display:].sum()
            if other_count > 0:
                pct_other = other_count / n_total * 100
                output.append(f"  [Other]:      {other_count:8,}  ({pct_other:6.2f}%)")
            break
        
        pct = count / n_total * 100
        # Truncate category name if too long
        cat_str = str(category)[:40]
        output.append(f"  {cat_str:40s}  {count:8,}  ({pct:6.2f}%)")
    
    return "\n".join(output)


def format_continuous_variable(series, var_name, var_info, var_description=None):
    """
    Format summary statistics for continuous variables as text.
    
    Continuous variables have many possible values, like:
    - Age (15-49 years for DHS)
    - Number of children
    - Household wealth index
    - Sampling weights
    
    We show:
    - Key statistics: mean, median, standard deviation, range
    - Quartile information
    - Missing data count
    
    Parameters:
    -----------
    series : pd.Series
        The variable data (should be numeric)
    var_name : str
        Name of the variable (for the header)
    var_info : dict
        Dictionary with pre-computed statistics from identify_variable_type()
    var_description : str, optional
        Human-readable description of the variable
    
    Returns:
    --------
    str : Formatted text summary
    """
    # Build the text output
    output = []
    output.append(f"VARIABLE: {var_name}")
    if var_description:
        output.append(f"DESCRIPTION: {var_description}")
    output.append(f"TYPE: Continuous/Numeric ({var_info['n_unique']} unique values)")
    output.append(f"TOTAL RESPONSES: {len(series):,}")
    output.append("")
    output.append("SUMMARY STATISTICS:")
    output.append(f"  Valid responses:    {var_info['n_valid']:12,}")
    output.append(f"  Missing:            {var_info['n_missing']:12,}")
    output.append(f"  Mean:               {var_info['mean']:12.4f}")
    output.append(f"  Median:             {var_info['median']:12.4f}")
    output.append(f"  Std. Deviation:     {var_info['std']:12.4f}")
    output.append(f"  Minimum:            {var_info['min']:12.4f}")
    output.append(f"  25th Percentile:    {var_info['q25']:12.4f}")
    output.append(f"  75th Percentile:    {var_info['q75']:12.4f}")
    output.append(f"  Maximum:            {var_info['max']:12.4f}")
    
    return "\n".join(output)


def format_identifier_variable(series, var_name, var_description=None):
    """
    Format summary for identifier variables (like case IDs) as text.
    
    Identifiers are not meaningful to analyze statistically.
    Instead, we just show basic info:
    - How many unique IDs exist
    - Whether there are duplicates (there shouldn't be!)
    
    Parameters:
    -----------
    series : pd.Series
        The identifier variable
    var_name : str
        Name of the variable (for the header)
    var_description : str, optional
        Human-readable description of the variable
    
    Returns:
    --------
    str : Formatted text summary
    """
    n_total = len(series)
    n_unique = series.nunique(dropna=True)
    n_missing = series.isna().sum()
    n_duplicates = n_total - n_missing - n_unique
    
    # Build the text output
    output = []
    output.append(f"VARIABLE: {var_name}")
    if var_description:
        output.append(f"DESCRIPTION: {var_description}")
    output.append(f"TYPE: Identifier")
    output.append(f"TOTAL RECORDS: {n_total:,}")
    output.append("")
    output.append("IDENTIFIER STATISTICS:")
    output.append(f"  Unique IDs:     {n_unique:12,}")
    output.append(f"  Missing:        {n_missing:12,}")
    output.append(f"  Duplicates:     {n_duplicates:12,}")
    
    if n_duplicates > 0:
        output.append("")
        output.append("  ⚠ WARNING: Duplicate IDs found!")
    
    return "\n".join(output)


def format_variable(series, var_name, var_descriptions=None):
    """
    Master formatting function that routes to the appropriate format type.
    
    This function:
    1. Identifies what type of variable it is
    2. Calls the appropriate specialized formatting function
    3. Returns formatted text
    
    Parameters:
    -----------
    series : pd.Series
        The variable data
    var_name : str
        Name of the variable (for the header)
    var_descriptions : dict, optional
        Dictionary mapping variable names to descriptions
    
    Returns:
    --------
    str : Formatted text summary
    """
    # Get the variable description if available
    var_description = None
    if var_descriptions and var_name.lower() in var_descriptions:
        var_description = var_descriptions[var_name.lower()]
    
    # Identify the variable type
    var_info = identify_variable_type(series)
    var_type = var_info['type']
    
    # Route to appropriate formatting function based on type
    if var_type == 'identifier':
        return format_identifier_variable(series, var_name, var_description)
    
    elif var_type == 'all_missing':
        output = []
        output.append(f"VARIABLE: {var_name}")
        if var_description:
            output.append(f"DESCRIPTION: {var_description}")
        output.append(f"TYPE: All Missing")
        output.append(f"TOTAL RESPONSES: {len(series):,}")
        output.append("")
        output.append("⚠ WARNING: All values are missing for this variable")
        return "\n".join(output)
    
    elif var_type == 'binary':
        return format_binary_variable(series, var_name, var_description)
    
    elif var_type == 'categorical':
        return format_categorical_variable(series, var_name, var_description)
    
    elif var_type == 'continuous':
        return format_continuous_variable(series, var_name, var_info, var_description)
    
    elif var_type == 'categorical_many':
        # For categorical with many levels, treat similar to regular categorical
        return format_categorical_variable(series, var_name, var_description)
    
    else:
        # Fallback for any unexpected type
        output = []
        output.append(f"VARIABLE: {var_name}")
        if var_description:
            output.append(f"DESCRIPTION: {var_description}")
        output.append(f"TYPE: Unknown ({var_type})")
        output.append(f"TOTAL RESPONSES: {len(series):,}")
        output.append("")
        output.append("Unable to generate summary statistics for this variable type")
        return "\n".join(output)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the summary statistics generation.
    """
    
    print("\n" + "=" * 70)
    print("DHS SUMMARY STATISTICS GENERATOR - DRC 2023-24 Survey")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/4] Verifying input files...")
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
    print("\n[STEP 2/4] Loading data...")
    print("-" * 70)
    
    print(f"\nReading: {INPUT_CSV.name}")
    data = pd.read_csv(INPUT_CSV)
    print(f"  → Loaded {len(data):,} women's records")
    print(f"  → {len(data.columns)} total variables")
    
    # Convert column names to lowercase for consistency
    data.columns = [col.lower() for col in data.columns]
    
    # Load variable descriptions (if available)
    print(f"\nLoading variable descriptions...")
    var_descriptions = load_variable_descriptions(VARIABLE_NAMES_MAP)
    
    # Get list of all variables
    all_variables = list(data.columns)
    print(f"\n✓ Will create summary statistics for {len(all_variables)} variables")
    
    # -------------------------------------------------------------------------
    # STEP 3: ANALYZE VARIABLE TYPES
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] Analyzing variable types...")
    print("-" * 70)
    
    # Count different types of variables
    type_counts = {
        'binary': 0,
        'categorical': 0,
        'continuous': 0,
        'identifier': 0,
        'all_missing': 0,
        'other': 0
    }
    
    print("\nScanning all variables...")
    for var in all_variables:
        var_info = identify_variable_type(data[var])
        var_type = var_info['type']
        
        if var_type in type_counts:
            type_counts[var_type] += 1
        else:
            type_counts['other'] += 1
    
    print("\nVariable type distribution:")
    for var_type, count in type_counts.items():
        if count > 0:
            print(f"  • {var_type:20s}: {count:4d} variables")
    
    # -------------------------------------------------------------------------
    # STEP 4: GENERATE SUMMARY STATISTICS
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Generating summary statistics...")
    print("-" * 70)
    
    # Create the output text file
    output_txt = OUTPUT_DIR / "2d_summary_statistics.txt"
    
    print(f"\nWriting summary statistics to: {output_txt.name}")
    
    with open(output_txt, 'w', encoding='utf-8') as f:
        # Write header
        f.write(SECTION_CHAR * LINE_WIDTH + "\n")
        f.write("DHS SUMMARY STATISTICS - DRC 2023-24 Survey\n")
        f.write(SECTION_CHAR * LINE_WIDTH + "\n")
        f.write("\n")
        f.write(f"Generated from: {INPUT_CSV.name}\n")
        f.write(f"Total women surveyed: {len(data):,}\n")
        f.write(f"Total variables: {len(all_variables)}\n")
        f.write("\n")
        f.write("Variable Type Distribution:\n")
        for var_type, count in type_counts.items():
            if count > 0:
                f.write(f"  • {var_type:20s}: {count:4d} variables\n")
        f.write("\n")
        f.write(SECTION_CHAR * LINE_WIDTH + "\n")
        f.write("\n\n")
        
        # Process each variable
        for i, var in enumerate(all_variables, 1):
            print(f"  Processing variable {i}/{len(all_variables)}: {var}")
            
            # Generate the summary for this variable
            var_summary = format_variable(data[var], var, var_descriptions)
            
            # Write to file
            f.write(SUBSECTION_CHAR * LINE_WIDTH + "\n")
            f.write(f"VARIABLE {i} of {len(all_variables)}\n")
            f.write(SUBSECTION_CHAR * LINE_WIDTH + "\n")
            f.write("\n")
            f.write(var_summary)
            f.write("\n\n\n")
    
    print(f"\n✓ Summary statistics written to: {output_txt.name}")
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 OUTPUT FILE (saved in {OUTPUT_DIR.name}/):\n")
    
    print(f"  {output_txt.name}")
    print(f"  → Easy-to-read text file with summary statistics")
    print(f"  → All {len(all_variables)} variables included")
    
    print(f"\n📈 VARIABLE SUMMARY:")
    print(f"  → Total variables: {len(all_variables)}")
    print(f"  → Binary (Yes/No): {type_counts['binary']}")
    print(f"  → Categorical: {type_counts['categorical']}")
    print(f"  → Continuous: {type_counts['continuous']}")
    print(f"  → Identifiers: {type_counts['identifier']}")
    if type_counts['all_missing'] > 0:
        print(f"  → All missing: {type_counts['all_missing']}")
    
    print("\n✅ Summary statistics generated successfully!\n")


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
