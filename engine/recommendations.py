"""
AARHI - Policy Recommendations Generation Module

Generates clear, rule-based, explainable recommendations for each district
based on observed resilience patterns and hotspot classifications.

Recommendations are designed to be:
- Transparent and rule-based (no ML or statistical inference)
- Human-readable and suitable for dashboards/reports
- Policy-ready with neutral, administrative tone

Important Notes:
- No causal or diagnostic language is used
- All findings are treated as observed proxy indicators
- Recommendations focus on suggested administrative interventions
"""

import pandas as pd
from pathlib import Path


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hotspot_data(filepath: str = "data/processed/hotspot_clusters.csv") -> pd.DataFrame:
    """
    Load clustered district data with hotspot labels.
    
    Args:
        filepath: Relative path to the hotspot clusters CSV file
        
    Returns:
        DataFrame with hotspot classifications and proxy ratios
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} district records from {filepath}")
    print(f"Columns: {list(df.columns)}")
    return df


# =============================================================================
# RECOMMENDATION RULES
# =============================================================================

def determine_dominant_indicator(row: pd.Series) -> str:
    """
    Determine which update indicator is dominant for a district.
    
    Compares biometric and demographic update ratios to identify
    which type of update activity is relatively more prevalent.
    
    Args:
        row: Series containing biometric_update_ratio and demographic_update_ratio
        
    Returns:
        String indicating dominant indicator: "biometric", "demographic", or "balanced"
    """
    bio_ratio = row.get("biometric_update_ratio", 0)
    demo_ratio = row.get("demographic_update_ratio", 0)
    
    # Handle edge case where both are zero or very small
    if bio_ratio < 0.01 and demo_ratio < 0.01:
        return "balanced"
    
    # Compare ratios with a 20% threshold for dominance
    if bio_ratio > demo_ratio * 1.2:
        return "biometric"
    elif demo_ratio > bio_ratio * 1.2:
        return "demographic"
    else:
        return "balanced"


def generate_red_recommendation(dominant_indicator: str) -> tuple:
    """
    Generate recommendation for Red (High Stress) hotspots.
    
    Args:
        dominant_indicator: "biometric", "demographic", or "balanced"
        
    Returns:
        Tuple of (recommendation text, basis text)
    """
    if dominant_indicator == "biometric":
        recommendation = (
            "Suggested administrative intervention: Consider biometric re-enrolment support, "
            "device calibration review, or operator training programs in this district."
        )
        basis = "Observed elevated biometric update intensity relative to enrolment volume."
        
    elif dominant_indicator == "demographic":
        recommendation = (
            "Suggested administrative intervention: Consider process review, public awareness "
            "campaigns, or data quality audits to address observed demographic update patterns."
        )
        basis = "Observed elevated demographic update intensity relative to enrolment volume."
        
    else:  # balanced
        recommendation = (
            "Suggested administrative intervention: Comprehensive operational review recommended "
            "covering both biometric and demographic update processes. Consider resource "
            "augmentation and capacity assessment."
        )
        basis = "Observed elevated intensity across both biometric and demographic update activities."
    
    return recommendation, basis


def generate_yellow_recommendation(dominant_indicator: str) -> tuple:
    """
    Generate recommendation for Yellow (Moderate Stress) hotspots.
    
    Args:
        dominant_indicator: "biometric", "demographic", or "balanced"
        
    Returns:
        Tuple of (recommendation text, basis text)
    """
    if dominant_indicator == "biometric":
        recommendation = (
            "Targeted monitoring recommended: Track biometric update trends and consider "
            "preventive measures such as equipment maintenance schedules or operator refresher training."
        )
        basis = "Observed moderate biometric update intensity indicative of potential operational stress."
        
    elif dominant_indicator == "demographic":
        recommendation = (
            "Targeted monitoring recommended: Track demographic update trends and consider "
            "preventive outreach or periodic data quality reviews in this district."
        )
        basis = "Observed moderate demographic update intensity indicative of potential operational stress."
        
    else:  # balanced
        recommendation = (
            "Periodic review recommended: Monitor both biometric and demographic update trends. "
            "Consider proactive capacity planning and preventive interventions."
        )
        basis = "Observed moderate intensity across update activities suggesting watchlist status."
    
    return recommendation, basis


def generate_green_recommendation(dominant_indicator: str) -> tuple:
    """
    Generate recommendation for Green (Stable) hotspots.
    
    Args:
        dominant_indicator: "biometric", "demographic", or "balanced"
        
    Returns:
        Tuple of (recommendation text, basis text)
    """
    if dominant_indicator == "biometric":
        recommendation = (
            "Continue standard monitoring. District shows stable patterns. Consider documenting "
            "operational practices for potential replication in higher-stress areas."
        )
        basis = "Observed stable operational patterns with manageable biometric update activity."
        
    elif dominant_indicator == "demographic":
        recommendation = (
            "Continue standard monitoring. District shows stable patterns. Document demographic "
            "update handling practices for potential best-practice sharing."
        )
        basis = "Observed stable operational patterns with manageable demographic update activity."
        
    else:  # balanced
        recommendation = (
            "Continue standard monitoring. District demonstrates stable and balanced operational "
            "patterns. Consider as a model for best-practice documentation and knowledge sharing."
        )
        basis = "Observed stable and balanced operational patterns across all indicators."
    
    return recommendation, basis


# =============================================================================
# RECOMMENDATION GENERATION
# =============================================================================

def generate_recommendation(row: pd.Series) -> tuple:
    """
    Generate a recommendation for a single district based on rule-based logic.
    
    Rules are based on:
    1. Hotspot label (Red / Yellow / Green)
    2. Relative magnitude of biometric vs demographic update ratios
    
    Args:
        row: Series containing district data with hotspot_label and ratios
        
    Returns:
        Tuple of (recommendation text, basis text)
    """
    hotspot_label = row.get("hotspot_label", "Green")
    dominant_indicator = determine_dominant_indicator(row)
    
    if hotspot_label == "Red":
        return generate_red_recommendation(dominant_indicator)
    elif hotspot_label == "Yellow":
        return generate_yellow_recommendation(dominant_indicator)
    else:  # Green or unknown
        return generate_green_recommendation(dominant_indicator)


def apply_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply recommendation rules to all districts.
    
    Generates recommendation and basis columns for each district
    using transparent, rule-based logic.
    
    Args:
        df: DataFrame with hotspot labels and proxy ratios
        
    Returns:
        DataFrame with recommendation and basis columns added
    """
    df = df.copy()
    
    # Generate recommendations for each row
    recommendations = []
    bases = []
    
    for idx, row in df.iterrows():
        rec, basis = generate_recommendation(row)
        recommendations.append(rec)
        bases.append(basis)
    
    df["recommendation"] = recommendations
    df["basis"] = bases
    
    # Print summary
    print(f"Generated recommendations for {len(df)} districts")
    print(f"\nRecommendation distribution by hotspot label:")
    for label in ["Red", "Yellow", "Green"]:
        count = (df["hotspot_label"] == label).sum()
        if count > 0:
            print(f"  {label}: {count} districts")
    
    return df


# =============================================================================
# OUTPUT
# =============================================================================

def prepare_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the final output DataFrame with selected columns.
    
    Output columns:
    - state
    - district
    - hotspot_label
    - recommendation
    - basis
    
    Args:
        df: DataFrame with all columns including recommendations
        
    Returns:
        DataFrame with only the required output columns
    """
    output_cols = ["state", "district", "hotspot_label", "recommendation", "basis"]
    
    # Verify all columns exist
    missing_cols = [col for col in output_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    output_df = df[output_cols].copy()
    
    # Sort by hotspot priority (Red first, then Yellow, then Green)
    label_priority = {"Red": 0, "Yellow": 1, "Green": 2}
    output_df["_sort_priority"] = output_df["hotspot_label"].map(label_priority)
    output_df = output_df.sort_values("_sort_priority").drop(columns=["_sort_priority"])
    output_df = output_df.reset_index(drop=True)
    
    return output_df


def save_recommendations(df: pd.DataFrame, output_path: str = "data/processed/recommendations.csv") -> None:
    """
    Save recommendations to CSV file.
    
    Args:
        df: DataFrame with recommendations
        output_path: Relative path for output CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved recommendations to: {output_path}")
    print(f"  Total districts: {len(df)}")
    print(f"  Columns: {list(df.columns)}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Execute the full recommendations generation pipeline.
    
    Steps:
    1. Load hotspot-clustered district data
    2. Apply rule-based recommendation logic
    3. Prepare output with required columns
    4. Save recommendations to CSV
    """
    print("=" * 70)
    print("AARHI - Policy Recommendations Generation")
    print("=" * 70)
    
    try:
        # -----------------------------------------------------------------
        # Step 1: Load hotspot data
        # -----------------------------------------------------------------
        print("\n[1/4] Loading hotspot-clustered district data...")
        df = load_hotspot_data()
        
        # -----------------------------------------------------------------
        # Step 2: Apply recommendation rules
        # -----------------------------------------------------------------
        print("\n[2/4] Applying rule-based recommendations...")
        df_with_recs = apply_recommendations(df)
        
        # -----------------------------------------------------------------
        # Step 3: Prepare output
        # -----------------------------------------------------------------
        print("\n[3/4] Preparing output dataset...")
        output_df = prepare_output_dataframe(df_with_recs)
        
        # -----------------------------------------------------------------
        # Step 4: Save results
        # -----------------------------------------------------------------
        print("\n[4/4] Saving recommendations...")
        save_recommendations(output_df)
        
        # Print sample recommendations
        print("\n" + "-" * 70)
        print("Sample recommendations by hotspot label:")
        for label in ["Red", "Yellow", "Green"]:
            sample = output_df[output_df["hotspot_label"] == label].head(1)
            if not sample.empty:
                row = sample.iloc[0]
                print(f"\n{label} Example ({row['state']} - {row['district']}):")
                print(f"  Recommendation: {row['recommendation'][:100]}...")
                print(f"  Basis: {row['basis']}")
        
        print("\n" + "=" * 70)
        print("Recommendations generation pipeline completed successfully!")
        print("=" * 70)
        
        return output_df
        
    except FileNotFoundError as e:
        print(f"\nError: Required data file not found.")
        print(f"Details: {e}")
        print("Please ensure clustering.py has been run first.")
        raise
    except Exception as e:
        print(f"\nError during recommendations generation: {e}")
        raise


if __name__ == "__main__":
    main()
