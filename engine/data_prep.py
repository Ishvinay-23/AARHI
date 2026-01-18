"""
AARHI Data Preparation Module

Loads, cleans, harmonizes, and merges three UIDAI-provided datasets:
- Aadhaar Enrolment dataset
- Aadhaar Demographic Update dataset
- Aadhaar Biometric Update dataset

Treats enrolment and update counts as proxy indicators of operational stress.
No inference of authentication failures or causal claims.
"""

import pandas as pd
from pathlib import Path
from typing import List


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_csvs_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a folder (including subfolders).
    
    Args:
        folder_path: Relative path to the folder containing CSV files
        
    Returns:
        Concatenated DataFrame from all CSV files
    """
    base_path = Path(folder_path)
    
    # Find all CSV files recursively (only actual files, not directories)
    csv_files = [f for f in base_path.glob("**/*.csv") if f.is_file()]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    print(f"  Found {len(csv_files)} CSV files in {folder_path}")
    
    # Load and concatenate all CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"    Loaded {csv_file.name}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total combined rows: {len(combined)}")
    
    return combined


def load_biometric_data() -> pd.DataFrame:
    """
    Load all Aadhaar Biometric Update CSV files.
    
    Returns:
        DataFrame with all biometric update data
    """
    return load_csvs_from_folder("data/raw/api_data_aadhar_biometric")


def load_demographic_data() -> pd.DataFrame:
    """
    Load all Aadhaar Demographic Update CSV files.
    
    Returns:
        DataFrame with all demographic update data
    """
    return load_csvs_from_folder("data/raw/api_data_aadhar_demographic")


def load_enrolment_data() -> pd.DataFrame:
    """
    Load all Aadhaar Enrolment CSV files.
    
    Returns:
        DataFrame with all enrolment data
    """
    return load_csvs_from_folder("data/raw/api_data_aadhar_enrolment")


# =============================================================================
# DATA STANDARDIZATION FUNCTIONS
# =============================================================================

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase with underscores.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def standardize_dates(df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
    """
    Convert date columns to datetime format.
    
    Args:
        df: Input DataFrame
        date_columns: List of column names that contain dates
        
    Returns:
        DataFrame with standardized date columns
    """
    df = df.copy()
    for col in date_columns:
        if col in df.columns:
            # Handle common date formats (DD-MM-YYYY, YYYY-MM-DD, etc.)
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows, keeping the first occurrence.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_rows = len(df)
    df = df.drop_duplicates(keep="first")
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"  Removed {removed} duplicate rows")
    return df


def remove_missing_critical(df: pd.DataFrame, critical_cols: List[str]) -> pd.DataFrame:
    """
    Remove rows with missing values in critical columns.
    
    Args:
        df: Input DataFrame
        critical_cols: List of column names that must not have missing values
        
    Returns:
        DataFrame with rows containing missing critical values removed
    """
    initial_rows = len(df)
    existing_cols = [col for col in critical_cols if col in df.columns]
    if existing_cols:
        df = df.dropna(subset=existing_cols)
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"  Removed {removed} rows with missing critical values")
    return df


def clean_invalid_values(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Clean invalid numeric values (negative counts, non-numeric entries).
    
    Args:
        df: Input DataFrame
        numeric_cols: List of columns that should be numeric
        
    Returns:
        DataFrame with invalid values cleaned
    """
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Replace negative values with NaN
            df.loc[df[col] < 0, col] = pd.NA
            # Fill NaN with 0 for count columns
            df[col] = df[col].fillna(0).astype(int)
    return df


def clean_state_district_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate state/district names.
    Removes rows with numeric-only or empty state/district values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned state/district names
    """
    df = df.copy()
    initial_rows = len(df)
    
    # Convert to string and strip whitespace
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip()
    if "district" in df.columns:
        df["district"] = df["district"].astype(str).str.strip()
    
    # Remove rows where state or district is numeric-only, empty, or 'nan'
    invalid_patterns = ["", "nan", "none", "null"]
    
    if "state" in df.columns:
        # Filter out invalid state values
        df = df[~df["state"].str.lower().isin(invalid_patterns)]
        # Filter out numeric-only states
        df = df[~df["state"].str.match(r"^\d+$", na=False)]
    
    if "district" in df.columns:
        # Filter out invalid district values
        df = df[~df["district"].str.lower().isin(invalid_patterns)]
        # Filter out numeric-only districts
        df = df[~df["district"].str.match(r"^\d+$", na=False)]
    
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"  Removed {removed} rows with invalid state/district names")
    
    return df


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def clean_enrolment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Aadhaar Enrolment dataset.
    
    Columns expected: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
    
    Args:
        df: Raw enrolment DataFrame
        
    Returns:
        Cleaned enrolment DataFrame with total_enrolments computed
    """
    print("\n  Cleaning enrolment data...")
    
    df = standardize_columns(df)
    df = standardize_dates(df, ["date"])
    df = remove_duplicates(df)
    df = remove_missing_critical(df, ["state", "district", "date"])
    df = clean_state_district_names(df)
    
    # Clean numeric columns
    age_cols = ["age_0_5", "age_5_17", "age_18_greater"]
    df = clean_invalid_values(df, age_cols)
    
    # Compute total enrolments (sum of all age groups)
    existing_age_cols = [col for col in age_cols if col in df.columns]
    if existing_age_cols:
        df["total_enrolments"] = df[existing_age_cols].sum(axis=1)
    else:
        df["total_enrolments"] = 0
    
    print(f"  Cleaned enrolment records: {len(df)}")
    return df


def clean_demographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Aadhaar Demographic Update dataset.
    
    Columns expected: date, state, district, pincode, demo_age_5_17, demo_age_17_
    
    Args:
        df: Raw demographic update DataFrame
        
    Returns:
        Cleaned demographic update DataFrame with total_demographic_updates computed
    """
    print("\n  Cleaning demographic update data...")
    
    df = standardize_columns(df)
    df = standardize_dates(df, ["date"])
    df = remove_duplicates(df)
    df = remove_missing_critical(df, ["state", "district", "date"])
    df = clean_state_district_names(df)
    
    # Clean numeric columns
    demo_cols = ["demo_age_5_17", "demo_age_17_"]
    df = clean_invalid_values(df, demo_cols)
    
    # Compute total demographic updates (sum of all age groups)
    existing_demo_cols = [col for col in demo_cols if col in df.columns]
    if existing_demo_cols:
        df["total_demographic_updates"] = df[existing_demo_cols].sum(axis=1)
    else:
        df["total_demographic_updates"] = 0
    
    print(f"  Cleaned demographic update records: {len(df)}")
    return df


def clean_biometric_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Aadhaar Biometric Update dataset.
    
    Columns expected: date, state, district, pincode, bio_age_5_17, bio_age_17_
    
    Args:
        df: Raw biometric update DataFrame
        
    Returns:
        Cleaned biometric update DataFrame with total_biometric_updates computed
    """
    print("\n  Cleaning biometric update data...")
    
    df = standardize_columns(df)
    df = standardize_dates(df, ["date"])
    df = remove_duplicates(df)
    df = remove_missing_critical(df, ["state", "district", "date"])
    df = clean_state_district_names(df)
    
    # Clean numeric columns
    bio_cols = ["bio_age_5_17", "bio_age_17_"]
    df = clean_invalid_values(df, bio_cols)
    
    # Compute total biometric updates (sum of all age groups)
    existing_bio_cols = [col for col in bio_cols if col in df.columns]
    if existing_bio_cols:
        df["total_biometric_updates"] = df[existing_bio_cols].sum(axis=1)
    else:
        df["total_biometric_updates"] = 0
    
    print(f"  Cleaned biometric update records: {len(df)}")
    return df


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_enrolment_by_district_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate enrolment data at state × district × date level.
    
    Args:
        df: Cleaned enrolment DataFrame
        
    Returns:
        Aggregated DataFrame with total_enrolments per state-district-date
    """
    groupby_cols = ["state", "district", "date"]
    
    agg_df = (
        df.groupby(groupby_cols, as_index=False)
        .agg({"total_enrolments": "sum"})
    )
    
    print(f"  Aggregated enrolment: {len(agg_df)} state-district-date combinations")
    return agg_df


def aggregate_demographic_by_district_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate demographic update data at state × district × date level.
    
    Args:
        df: Cleaned demographic update DataFrame
        
    Returns:
        Aggregated DataFrame with total_demographic_updates per state-district-date
    """
    groupby_cols = ["state", "district", "date"]
    
    agg_df = (
        df.groupby(groupby_cols, as_index=False)
        .agg({"total_demographic_updates": "sum"})
    )
    
    print(f"  Aggregated demographic updates: {len(agg_df)} state-district-date combinations")
    return agg_df


def aggregate_biometric_by_district_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate biometric update data at state × district × date level.
    
    Args:
        df: Cleaned biometric update DataFrame
        
    Returns:
        Aggregated DataFrame with total_biometric_updates per state-district-date
    """
    groupby_cols = ["state", "district", "date"]
    
    agg_df = (
        df.groupby(groupby_cols, as_index=False)
        .agg({"total_biometric_updates": "sum"})
    )
    
    print(f"  Aggregated biometric updates: {len(agg_df)} state-district-date combinations")
    return agg_df


# =============================================================================
# MERGE FUNCTIONS
# =============================================================================

def merge_datasets(
    enrolment_df: pd.DataFrame,
    demographic_df: pd.DataFrame,
    biometric_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge three aggregated datasets on state, district, and date.
    
    Args:
        enrolment_df: Aggregated enrolment DataFrame
        demographic_df: Aggregated demographic update DataFrame
        biometric_df: Aggregated biometric update DataFrame
        
    Returns:
        Merged DataFrame with all metrics combined
    """
    merge_cols = ["state", "district", "date"]
    
    # Merge enrolment with demographic updates
    merged = enrolment_df.merge(
        demographic_df,
        on=merge_cols,
        how="outer"
    )
    
    # Merge with biometric updates
    merged = merged.merge(
        biometric_df,
        on=merge_cols,
        how="outer"
    )
    
    # Fill NaN counts with 0
    count_cols = ["total_enrolments", "total_demographic_updates", "total_biometric_updates"]
    for col in count_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
    
    # Sort by state, district, date for readability
    merged = merged.sort_values(by=merge_cols).reset_index(drop=True)
    
    print(f"\n  Merged dataset: {len(merged)} total records")
    return merged


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_processed_data(df: pd.DataFrame, output_filename: str = "district_merged_metrics.csv") -> None:
    """
    Save processed DataFrame to data/processed/ directory.
    
    Args:
        df: DataFrame to save
        output_filename: Name of output CSV file
    """
    output_path = Path("data") / "processed" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n  Processed data saved to: {output_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Execute the complete data preparation pipeline.
    
    Steps:
    1. Load all CSV files from each dataset folder
    2. Concatenate chunked CSVs per dataset
    3. Standardize column names and dates
    4. Clean missing, duplicate, and invalid rows
    5. Compute total counts per dataset
    6. Aggregate data at state × district × date level
    7. Merge the three datasets
    8. Save to data/processed/district_merged_metrics.csv
    """
    print("=" * 70)
    print("AARHI Data Preparation Pipeline")
    print("=" * 70)
    
    try:
        # -----------------------------------------------------------------
        # Step 1: Load raw datasets from chunked CSV folders
        # -----------------------------------------------------------------
        print("\n[1/6] Loading raw datasets...")
        
        print("\nLoading Biometric Update data:")
        biometric_raw = load_biometric_data()
        
        print("\nLoading Demographic Update data:")
        demographic_raw = load_demographic_data()
        
        print("\nLoading Enrolment data:")
        enrolment_raw = load_enrolment_data()
        
        # -----------------------------------------------------------------
        # Step 2: Clean and compute totals for each dataset
        # -----------------------------------------------------------------
        print("\n[2/6] Cleaning and processing datasets...")
        
        enrolment_clean = clean_enrolment_data(enrolment_raw)
        demographic_clean = clean_demographic_data(demographic_raw)
        biometric_clean = clean_biometric_data(biometric_raw)
        
        # -----------------------------------------------------------------
        # Step 3: Aggregate at state × district × date level
        # -----------------------------------------------------------------
        print("\n[3/6] Aggregating at state × district × date level...")
        
        enrolment_agg = aggregate_enrolment_by_district_date(enrolment_clean)
        demographic_agg = aggregate_demographic_by_district_date(demographic_clean)
        biometric_agg = aggregate_biometric_by_district_date(biometric_clean)
        
        # -----------------------------------------------------------------
        # Step 4: Merge the three aggregated datasets
        # -----------------------------------------------------------------
        print("\n[4/6] Merging datasets on state, district, date...")
        
        merged = merge_datasets(enrolment_agg, demographic_agg, biometric_agg)
        
        # -----------------------------------------------------------------
        # Step 5: Display summary statistics
        # -----------------------------------------------------------------
        print("\n[5/6] Summary statistics:")
        print(f"  Unique states: {merged['state'].nunique()}")
        print(f"  Unique districts: {merged['district'].nunique()}")
        print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
        print(f"  Total enrolments: {merged['total_enrolments'].sum():,}")
        print(f"  Total demographic updates: {merged['total_demographic_updates'].sum():,}")
        print(f"  Total biometric updates: {merged['total_biometric_updates'].sum():,}")
        
        # -----------------------------------------------------------------
        # Step 6: Save processed data
        # -----------------------------------------------------------------
        print("\n[6/6] Saving processed data...")
        
        save_processed_data(merged)
        
        print("\n" + "=" * 70)
        print("Data preparation pipeline completed successfully!")
        print("=" * 70)
        
        return merged
        
    except FileNotFoundError as e:
        print(f"\nError: Required data file not found.")
        print(f"Details: {e}")
        raise
    except Exception as e:
        print(f"\nError during data preparation: {e}")
        raise


if __name__ == "__main__":
    main()
