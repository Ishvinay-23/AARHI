"""
AARHI - Aadhaar Resilience Index (ARI) Computation Module

Computes a district-level Aadhaar Resilience Index using proxy-based indicators
derived from UIDAI-provided aggregated datasets.

The ARI is a composite score (0-100) that reflects operational patterns
based on enrolment and update activity ratios. Higher scores indicate
more stable operational conditions.

Important Notes:
- Update ratios are treated strictly as proxy indicators of operational stress
- No inference of authentication failures is made
- The methodology is transparent and suitable for non-technical review
"""

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# DATA LOADING
# =============================================================================

def load_merged_metrics(filepath: str = "data/processed/district_merged_metrics.csv") -> pd.DataFrame:
    """
    Load the merged district-level metrics from CSV.
    
    Args:
        filepath: Relative path to the merged metrics CSV file
        
    Returns:
        DataFrame with district-level metrics
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records from {filepath}")
    print(f"Columns: {list(df.columns)}")
    return df


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_by_district(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data at the district level by summing metrics across all dates.
    
    Groups by state and district, then sums enrolment and update counts.
    
    Args:
        df: DataFrame with date-level metrics
        
    Returns:
        DataFrame aggregated at state-district level
    """
    groupby_cols = ["state", "district"]
    metric_cols = ["total_enrolments", "total_demographic_updates", "total_biometric_updates"]
    
    # Ensure metric columns exist
    available_metrics = [col for col in metric_cols if col in df.columns]
    
    # Aggregate by summing across all dates
    agg_df = (
        df.groupby(groupby_cols, as_index=False)[available_metrics]
        .sum()
    )
    
    print(f"Aggregated to {len(agg_df)} unique state-district combinations")
    return agg_df


# =============================================================================
# RATIO COMPUTATION
# =============================================================================

def compute_proxy_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute proxy ratios that indicate operational patterns.
    
    Ratios computed:
    - biometric_update_ratio: total_biometric_updates / total_enrolments
    - demographic_update_ratio: total_demographic_updates / total_enrolments
    
    These ratios serve as proxy indicators of operational activity patterns.
    Higher ratios may indicate areas with more update activity relative to
    enrolment volume.
    
    Args:
        df: DataFrame with aggregated district metrics
        
    Returns:
        DataFrame with computed ratio columns added
    """
    df = df.copy()
    
    # Handle division by zero safely using np.where
    # If enrolments is 0, set ratio to 0 (no activity baseline)
    enrolments = df["total_enrolments"].values
    
    # Biometric update ratio
    df["biometric_update_ratio"] = np.where(
        enrolments > 0,
        df["total_biometric_updates"] / enrolments,
        0.0
    )
    
    # Demographic update ratio
    df["demographic_update_ratio"] = np.where(
        enrolments > 0,
        df["total_demographic_updates"] / enrolments,
        0.0
    )
    
    print(f"Computed proxy ratios:")
    print(f"  Biometric update ratio - min: {df['biometric_update_ratio'].min():.4f}, "
          f"max: {df['biometric_update_ratio'].max():.4f}, "
          f"mean: {df['biometric_update_ratio'].mean():.4f}")
    print(f"  Demographic update ratio - min: {df['demographic_update_ratio'].min():.4f}, "
          f"max: {df['demographic_update_ratio'].max():.4f}, "
          f"mean: {df['demographic_update_ratio'].mean():.4f}")
    
    return df


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization (0-1) to proxy ratios across all districts.
    
    Normalization formula: (value - min) / (max - min)
    
    If max equals min (all values identical), normalized value is set to 0.
    
    Args:
        df: DataFrame with computed ratios
        
    Returns:
        DataFrame with normalized ratio columns added
    """
    df = df.copy()
    
    # Normalize biometric update ratio
    bio_min = df["biometric_update_ratio"].min()
    bio_max = df["biometric_update_ratio"].max()
    bio_range = bio_max - bio_min
    
    if bio_range > 0:
        df["biometric_update_ratio_norm"] = (
            (df["biometric_update_ratio"] - bio_min) / bio_range
        )
    else:
        # All values are the same; set normalized to 0
        df["biometric_update_ratio_norm"] = 0.0
    
    # Normalize demographic update ratio
    demo_min = df["demographic_update_ratio"].min()
    demo_max = df["demographic_update_ratio"].max()
    demo_range = demo_max - demo_min
    
    if demo_range > 0:
        df["demographic_update_ratio_norm"] = (
            (df["demographic_update_ratio"] - demo_min) / demo_range
        )
    else:
        # All values are the same; set normalized to 0
        df["demographic_update_ratio_norm"] = 0.0
    
    print(f"Applied min-max normalization:")
    print(f"  Biometric ratio range: [{bio_min:.4f}, {bio_max:.4f}]")
    print(f"  Demographic ratio range: [{demo_min:.4f}, {demo_max:.4f}]")
    
    return df


# =============================================================================
# ARI SCORE COMPUTATION
# =============================================================================

def compute_ari_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Aadhaar Resilience Index (ARI) score for each district.
    
    Formula:
        ARI = 100 * (
            0.6 * (1 - biometric_update_ratio_norm) +
            0.4 * (1 - demographic_update_ratio_norm)
        )
    
    Interpretation:
    - Higher normalized ratios indicate more update activity relative to enrolments
    - The formula inverts the ratios so that LOWER update activity yields HIGHER ARI
    - Weights: Biometric updates (60%), Demographic updates (40%)
    - Final score is clipped to the 0-100 range
    
    Args:
        df: DataFrame with normalized ratios
        
    Returns:
        DataFrame with ARI score column added
    """
    df = df.copy()
    
    # Weight assignments (transparent and adjustable)
    weight_biometric = 0.6
    weight_demographic = 0.4
    
    # Compute ARI using weighted linear formula
    # Lower normalized ratios (less update activity) result in higher ARI scores
    df["ari_score"] = 100 * (
        weight_biometric * (1 - df["biometric_update_ratio_norm"]) +
        weight_demographic * (1 - df["demographic_update_ratio_norm"])
    )
    
    # Clip to 0-100 range for safety
    df["ari_score"] = df["ari_score"].clip(lower=0, upper=100)
    
    print(f"Computed ARI scores:")
    print(f"  Min: {df['ari_score'].min():.2f}")
    print(f"  Max: {df['ari_score'].max():.2f}")
    print(f"  Mean: {df['ari_score'].mean():.2f}")
    print(f"  Median: {df['ari_score'].median():.2f}")
    
    return df


# =============================================================================
# RESILIENCE CATEGORIZATION
# =============================================================================

def assign_resilience_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign resilience categories based on ARI score thresholds.
    
    Categories:
    - ARI >= 70: "Stable" - Lower operational stress indicators
    - 40 <= ARI < 70: "Moderate Stress" - Medium operational stress indicators
    - ARI < 40: "High Stress" - Higher operational stress indicators
    
    These categories are based on proxy indicators and should not be
    interpreted as direct measures of system performance.
    
    Args:
        df: DataFrame with ARI scores
        
    Returns:
        DataFrame with resilience_category column added
    """
    df = df.copy()
    
    # Define category thresholds
    threshold_stable = 70
    threshold_moderate = 40
    
    # Assign categories using numpy select for efficiency
    conditions = [
        df["ari_score"] >= threshold_stable,
        df["ari_score"] >= threshold_moderate,
        df["ari_score"] < threshold_moderate
    ]
    categories = ["Stable", "Moderate Stress", "High Stress"]
    
    df["resilience_category"] = np.select(conditions, categories, default="Unknown")
    
    # Count districts in each category
    category_counts = df["resilience_category"].value_counts()
    print(f"Resilience category distribution:")
    for category in ["Stable", "Moderate Stress", "High Stress"]:
        count = category_counts.get(category, 0)
        pct = (count / len(df)) * 100
        print(f"  {category}: {count} districts ({pct:.1f}%)")
    
    return df


# =============================================================================
# OUTPUT
# =============================================================================

def save_ari_results(df: pd.DataFrame, output_path: str = "data/processed/ari_scored_districts.csv") -> None:
    """
    Save the ARI-scored district data to CSV.
    
    Output columns include:
    - state, district
    - total_enrolments, total_demographic_updates, total_biometric_updates
    - biometric_update_ratio, demographic_update_ratio
    - biometric_update_ratio_norm, demographic_update_ratio_norm
    - ari_score, resilience_category
    
    Args:
        df: DataFrame with ARI scores and categories
        output_path: Relative path for output CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Round numeric columns for cleaner output
    df = df.copy()
    numeric_cols = [
        "biometric_update_ratio", "demographic_update_ratio",
        "biometric_update_ratio_norm", "demographic_update_ratio_norm",
        "ari_score"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].round(4)
    
    # Sort by ARI score (ascending) to show high-stress areas first
    df = df.sort_values("ari_score", ascending=True).reset_index(drop=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved ARI results to: {output_path}")
    print(f"  Total districts: {len(df)}")
    print(f"  Columns: {list(df.columns)}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Execute the full ARI computation pipeline.
    
    Steps:
    1. Load merged district metrics
    2. Aggregate at district level (sum across dates)
    3. Compute proxy ratios (update counts / enrolments)
    4. Normalize ratios using min-max scaling
    5. Compute ARI score using weighted formula
    6. Assign resilience categories
    7. Save results to CSV
    """
    print("=" * 70)
    print("AARHI - Aadhaar Resilience Index (ARI) Computation")
    print("=" * 70)
    
    try:
        # -----------------------------------------------------------------
        # Step 1: Load merged metrics
        # -----------------------------------------------------------------
        print("\n[1/6] Loading merged district metrics...")
        df = load_merged_metrics()
        
        # -----------------------------------------------------------------
        # Step 2: Aggregate at district level
        # -----------------------------------------------------------------
        print("\n[2/6] Aggregating at district level...")
        df_agg = aggregate_by_district(df)
        
        # -----------------------------------------------------------------
        # Step 3: Compute proxy ratios
        # -----------------------------------------------------------------
        print("\n[3/6] Computing proxy ratios...")
        df_ratios = compute_proxy_ratios(df_agg)
        
        # -----------------------------------------------------------------
        # Step 4: Normalize ratios
        # -----------------------------------------------------------------
        print("\n[4/6] Normalizing ratios (min-max scaling)...")
        df_norm = normalize_ratios(df_ratios)
        
        # -----------------------------------------------------------------
        # Step 5: Compute ARI score
        # -----------------------------------------------------------------
        print("\n[5/6] Computing ARI scores...")
        df_ari = compute_ari_score(df_norm)
        
        # -----------------------------------------------------------------
        # Step 6: Assign resilience categories
        # -----------------------------------------------------------------
        print("\n[6/6] Assigning resilience categories...")
        df_final = assign_resilience_category(df_ari)
        
        # -----------------------------------------------------------------
        # Save results
        # -----------------------------------------------------------------
        print("\n" + "-" * 70)
        print("Saving results...")
        save_ari_results(df_final)
        
        print("\n" + "=" * 70)
        print("ARI computation pipeline completed successfully!")
        print("=" * 70)
        
        return df_final
        
    except FileNotFoundError as e:
        print(f"\nError: Required data file not found.")
        print(f"Details: {e}")
        print("Please ensure data_prep.py has been run first.")
        raise
    except Exception as e:
        print(f"\nError during ARI computation: {e}")
        raise


if __name__ == "__main__":
    main()
