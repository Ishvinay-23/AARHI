<!--
1. Data Cleaning and Preprocessing
  •	Missing and invalid rows removed
  •	Column names standardized
  •	Dates normalized
  •	Data aggregated at district × date level
  •	Enrolment and update datasets merged into a unified table
This produced a consolidated district-level operational dataset.
________________________________________
2. Feature Engineering
To enable fair comparison across districts:
  •	Biometric Update Ratio = biometric updates / enrolments
  •	Demographic Update Ratio = demographic updates / enrolments
Ratios were preferred over raw counts to account for population scale differences. All ratios were normalized using min–max scaling.
________________________________________
3. Aadhaar Resilience Index (ARI)
The Aadhaar Resilience Index (ARI) is a composite score ranging from 0 to 100, representing relative operational stability.
ARI combines:
  •	Normalized biometric update intensity
  •	Normalized demographic update intensity
Higher ARI values indicate stable operational patterns, while lower values highlight relative stress. ARI is proxy-based, transparent, and non-causal.
“Representative code snippets are included below to demonstrate the analytical logic used. Complete source files may be shared separately if requested.”
Code Snapshot: Aadhaar Resilience Index (ARI) Computation
"""# Aadhaar Resilience Index (ARI) computation
df["ari_score"] = 100 * (
    0.6 * (1 - df["biometric_update_ratio_norm"]) +
    0.4 * (1 - df["demographic_update_ratio_norm"])
)"""

This logic combines normalized biometric and demographic update ratios into a single composite score to represent relative district-level operational resilience.
________________________________________
3.4 Hotspot Identification
Districts were grouped using K-Means clustering (k = 3) based on:
  • ARI score
  • Biometric update ratio
  • Demographic update ratio
Clusters were semantically labelled as:
  •	Red – High Stress
  •	Yellow – Moderate Stress
  •	Green – Stable
This enables interpretable geographic categorization.
Code Snapshot: District Hotspot Identification Using K-Means Clustering

"""from sklearn.cluster import KMeans
features = df[[
    "ari_score",
    "biometric_update_ratio",
    "demographic_update_ratio"
]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(features)"""

K-Means clustering is applied on ARI and update intensity features to group districts into high, moderate, and stable operational stress categories.
________________________________________
3.5 Recommendation Logic
A deterministic, rule-based system assigns recommendations:
  •	High Stress → targeted intervention guidance
  •	Moderate Stress → monitoring and preventive actions
  •	Stable → continuation and documentation
No black-box models are used.
________________________________________ -->
