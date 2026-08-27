"""
Phase 1: Dataset Inspection
Inspect AqSolDB to understand its structure, columns, and data types.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths - this works in VS Code
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "reports" / "dataset_inspection.txt"

# Make sure reports directory exists
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

def inspect_dataset():
    """Load and inspect the AqSolDB dataset."""
    
    print("=" * 70)
    print("AQSOLDB DATASET INSPECTION")
    print("=" * 70)
    
    # Find the CSV file
    csv_files = list(RAW_DATA_PATH.glob("*.csv"))
    if not csv_files:
        print("ERROR: No CSV file found in data/raw/")
        print(f"Files in {RAW_DATA_PATH}:")
        for f in RAW_DATA_PATH.iterdir():
            print(f"  - {f.name}")
        return None
    
    csv_file = csv_files[0]
    print(f"\n✓ Found data file: {csv_file.name}")
    
    # Load the dataset
    print("\nLoading dataset...")
    df = pd.read_csv(csv_file)
    print(f"   ✓ Loaded {len(df):,} rows")
    print(f"   ✓ Loaded {len(df.columns)} columns")
    
    # Column information
    print("\n" + "=" * 70)
    print("COLUMN INFORMATION")
    print("=" * 70)
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = len(df) - non_null
        null_pct = (null_count / len(df)) * 100
        print(f"   {i:2d}. {col:40s} | dtype: {str(dtype):10s} | nulls: {null_count:5d} ({null_pct:.1f}%)")
    
    # First few rows
    print("\n" + "=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)
    print(df.head().to_string())
    
    # Basic statistics for numeric columns
    print("\n" + "=" * 70)
    print("BASIC STATISTICS")
    print("=" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    
    # Check for duplicate SMILES
    if 'SMILES' in df.columns:
        print("\n" + "=" * 70)
        print("SMILES ANALYSIS")
        print("=" * 70)
        unique_smiles = df['SMILES'].nunique()
        print(f"   Unique SMILES: {unique_smiles:,}")
        print(f"   Duplicates: {len(df) - unique_smiles:,}")
    
    # Check for solubility-related columns
    solubility_cols = [col for col in df.columns if 'log' in col.lower() or 'sol' in col.lower()]
    if solubility_cols:
        print("\n" + "=" * 70)
        print("SOLUBILITY COLUMNS")
        print("=" * 70)
        for col in solubility_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                print(f"   {col}:")
                print(f"      Range: {df[col].min():.4f} to {df[col].max():.4f}")
                print(f"      Mean: {df[col].mean():.4f}")
                print(f"      Std: {df[col].std():.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Save report
    print(f"\n✓ Saving detailed report to: {REPORT_PATH}")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("AQSOLDB DATASET INSPECTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"File: {csv_file.name}\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Total columns: {len(df.columns)}\n\n")
        f.write("COLUMNS:\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"  {i}. {col}\n")
        f.write("\nDATA TYPES:\n")
        f.write(str(df.dtypes) + "\n")
        f.write("\nFIRST 5 ROWS:\n")
        f.write(df.head().to_string() + "\n")
        f.write("\n\nBASIC STATISTICS (numeric columns):\n")
        if len(numeric_cols) > 0:
            f.write(df[numeric_cols].describe().to_string())
    
    return df

if __name__ == "__main__":
    try:
        df = inspect_dataset()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()