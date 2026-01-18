"""
AARHI - Hotspot Identification and Clustering Module

Identifies district-level resilience hotspots by clustering districts based on
their Aadhaar Resilience Index (ARI) and proxy-based stress indicators.

Clustering is used to group districts with similar operational patterns.
Clusters are labeled as relative stress groupings (Red/Yellow/Green) for
easier interpretation by non-technical reviewers.

Important Notes:
- ARI values are NOT modified
- No causal inference is made
- Clusters represent relative stress groupings based on proxy indicators
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ari_data(filepath: str = "data/processed/ari_scored_districts.csv") -> pd.DataFrame:
    """
    Load ARI-scored district data from CSV.
    
    Args:
        filepath: Relative path to the ARI-scored districts CSV file
        
    Returns:
        DataFrame with ARI scores and proxy ratios
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} district records from {filepath}")
    print(f"Columns: {list(df.columns)}")
    return df


# =============================================================================
# DATA PREPARATION
# =============================================================================

def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select numeric, interpretable features for clustering.
    
    Features used:
    - ari_score: Overall resilience index
    - biometric_update_ratio: Biometric updates relative to enrolments
    - demographic_update_ratio: Demographic updates relative to enrolments
    
    Args:
        df: DataFrame with ARI data
        
    Returns:
        DataFrame containing only the clustering features
    """
    feature_cols = ["ari_score", "biometric_update_ratio", "demographic_update_ratio"]
    
    # Verify all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    features = df[feature_cols].copy()
    print(f"Selected {len(feature_cols)} features for clustering: {feature_cols}")
    
    return features


def handle_missing_values(df: pd.DataFrame, features: pd.DataFrame) -> tuple:
    """
    Handle missing values in the feature set safely.
    
    Strategy: Fill missing values with column median to avoid distorting
    the distribution. Records the indices of rows with missing values.
    
    Args:
        df: Original DataFrame (for tracking)
        features: DataFrame with clustering features
        
    Returns:
        Tuple of (cleaned features DataFrame, indices of imputed rows)
    """
    features = features.copy()
    
    # Track rows with missing values
    missing_mask = features.isna().any(axis=1)
    imputed_indices = df.index[missing_mask].tolist()
    
    if len(imputed_indices) > 0:
        print(f"Found {len(imputed_indices)} rows with missing values")
        
        # Fill missing values with column median
        for col in features.columns:
            if features[col].isna().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)
                print(f"  Filled missing values in '{col}' with median: {median_val:.4f}")
    else:
        print("No missing values found in clustering features")
    
    return features, imputed_indices


def scale_features(features: pd.DataFrame) -> tuple:
    """
    Apply standard scaling to features for K-Means clustering.
    
    K-Means is sensitive to feature scale, so standardization ensures
    all features contribute equally to distance calculations.
    
    Args:
        features: DataFrame with clustering features
        
    Returns:
        Tuple of (scaled features as numpy array, fitted scaler object)
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    print(f"Applied standard scaling to features")
    print(f"  Feature means after scaling: {scaled_features.mean(axis=0).round(6)}")
    print(f"  Feature stds after scaling: {scaled_features.std(axis=0).round(4)}")
    
    return scaled_features, scaler


# =============================================================================
# CLUSTERING
# =============================================================================

def apply_kmeans_clustering(scaled_features: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> np.ndarray:
    """
    Apply K-Means clustering to group districts by operational patterns.
    
    Uses K-Means with k=3 to create three distinct groups representing
    different levels of operational stress indicators.
    
    Args:
        scaled_features: Standardized feature array
        n_clusters: Number of clusters (default: 3)
        random_state: Random seed for reproducibility
        
    Returns:
        Array of cluster labels (0, 1, or 2)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,  # Number of initializations for stability
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    print(f"Applied K-Means clustering with k={n_clusters}")
    print(f"  Random state: {random_state} (deterministic)")
    print(f"  Cluster distribution:")
    for i in range(n_clusters):
        count = (cluster_labels == i).sum()
        pct = (count / len(cluster_labels)) * 100
        print(f"    Cluster {i}: {count} districts ({pct:.1f}%)")
    
    return cluster_labels


# =============================================================================
# CLUSTER LABELING
# =============================================================================

def map_clusters_to_labels(df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
    """
    Map numeric cluster IDs to semantic hotspot labels based on average ARI.
    
    Label assignment logic:
    - Cluster with HIGHEST average ARI → "Green" (Stable)
    - Cluster with MIDDLE average ARI → "Yellow" (Moderate Stress)
    - Cluster with LOWEST average ARI → "Red" (High Stress)
    
    This ensures labels are meaningful regardless of which cluster ID
    K-Means assigns to which group.
    
    Args:
        df: Original DataFrame with ARI scores
        cluster_labels: Array of cluster assignments
        
    Returns:
        DataFrame with cluster and hotspot_label columns added
    """
    df = df.copy()
    df["cluster"] = cluster_labels
    
    # Compute average ARI score per cluster
    cluster_avg_ari = df.groupby("cluster")["ari_score"].mean().sort_values()
    
    print(f"Average ARI score per cluster:")
    for cluster_id, avg_ari in cluster_avg_ari.items():
        print(f"  Cluster {cluster_id}: {avg_ari:.2f}")
    
    # Map clusters to labels based on ARI ranking
    # Lowest ARI → Red, Middle → Yellow, Highest → Green
    cluster_order = cluster_avg_ari.index.tolist()  # Sorted ascending by ARI
    
    label_mapping = {
        cluster_order[0]: "Red",     # Lowest ARI → High Stress
        cluster_order[1]: "Yellow",  # Middle ARI → Moderate Stress
        cluster_order[2]: "Green"    # Highest ARI → Stable
    }
    
    # Apply label mapping
    df["hotspot_label"] = df["cluster"].map(label_mapping)
    
    print(f"\nCluster to label mapping (based on average ARI):")
    for cluster_id, label in sorted(label_mapping.items()):
        print(f"  Cluster {cluster_id} → {label}")
    
    # Count districts per label
    print(f"\nHotspot label distribution:")
    label_counts = df["hotspot_label"].value_counts()
    for label in ["Red", "Yellow", "Green"]:
        count = label_counts.get(label, 0)
        pct = (count / len(df)) * 100
        print(f"  {label}: {count} districts ({pct:.1f}%)")
    
    return df


# =============================================================================
# OUTPUT
# =============================================================================

def save_clustered_results(df: pd.DataFrame, output_path: str = "data/processed/hotspot_clusters.csv") -> None:
    """
    Save the clustered district data to CSV.
    
    Output includes all original columns plus:
    - cluster: Numeric cluster ID (0, 1, 2)
    - hotspot_label: Semantic label (Red, Yellow, Green)
    
    Args:
        df: DataFrame with cluster assignments and labels
        output_path: Relative path for output CSV file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by hotspot label priority (Red first, then Yellow, then Green)
    label_priority = {"Red": 0, "Yellow": 1, "Green": 2}
    df = df.copy()
    df["_sort_priority"] = df["hotspot_label"].map(label_priority)
    df = df.sort_values(["_sort_priority", "ari_score"], ascending=[True, True])
    df = df.drop(columns=["_sort_priority"]).reset_index(drop=True)
    
    df.to_csv(output_file, index=False)
    
    print(f"\nSaved clustered results to: {output_path}")
    print(f"  Total districts: {len(df)}")
    print(f"  Columns: {list(df.columns)}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Execute the full hotspot clustering pipeline.
    
    Steps:
    1. Load ARI-scored district data
    2. Select clustering features (ari_score, biometric_update_ratio, demographic_update_ratio)
    3. Handle missing values (median imputation)
    4. Scale features (standard scaling)
    5. Apply K-Means clustering (k=3, deterministic)
    6. Map clusters to semantic labels (Red/Yellow/Green)
    7. Save results to CSV
    """
    print("=" * 70)
    print("AARHI - Hotspot Identification and Clustering")
    print("=" * 70)
    
    try:
        # -----------------------------------------------------------------
        # Step 1: Load ARI-scored data
        # -----------------------------------------------------------------
        print("\n[1/6] Loading ARI-scored district data...")
        df = load_ari_data()
        
        # -----------------------------------------------------------------
        # Step 2: Select clustering features
        # -----------------------------------------------------------------
        print("\n[2/6] Selecting clustering features...")
        features = select_clustering_features(df)
        
        # -----------------------------------------------------------------
        # Step 3: Handle missing values
        # -----------------------------------------------------------------
        print("\n[3/6] Handling missing values...")
        features_clean, imputed_indices = handle_missing_values(df, features)
        
        # -----------------------------------------------------------------
        # Step 4: Scale features
        # -----------------------------------------------------------------
        print("\n[4/6] Scaling features...")
        scaled_features, scaler = scale_features(features_clean)
        
        # -----------------------------------------------------------------
        # Step 5: Apply K-Means clustering
        # -----------------------------------------------------------------
        print("\n[5/6] Applying K-Means clustering (k=3)...")
        cluster_labels = apply_kmeans_clustering(scaled_features, n_clusters=3, random_state=42)
        
        # -----------------------------------------------------------------
        # Step 6: Map clusters to semantic labels
        # -----------------------------------------------------------------
        print("\n[6/6] Mapping clusters to hotspot labels...")
        df_clustered = map_clusters_to_labels(df, cluster_labels)
        
        # -----------------------------------------------------------------
        # Save results
        # -----------------------------------------------------------------
        print("\n" + "-" * 70)
        print("Saving results...")
        save_clustered_results(df_clustered)
        
        # Print summary statistics per hotspot
        print("\n" + "-" * 70)
        print("Summary statistics by hotspot label:")
        summary = df_clustered.groupby("hotspot_label").agg({
            "ari_score": ["mean", "min", "max"],
            "biometric_update_ratio": "mean",
            "demographic_update_ratio": "mean"
        }).round(2)
        print(summary.to_string())
        
        print("\n" + "=" * 70)
        print("Hotspot clustering pipeline completed successfully!")
        print("=" * 70)
        
        return df_clustered
        
    except FileNotFoundError as e:
        print(f"\nError: Required data file not found.")
        print(f"Details: {e}")
        print("Please ensure ari.py has been run first.")
        raise
    except Exception as e:
        print(f"\nError during clustering: {e}")
        raise


if __name__ == "__main__":
    main()
