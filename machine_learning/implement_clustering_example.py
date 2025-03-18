from machine_learning.create_cs_outputs_enriched_metadata import create_cs_outputs_enriched_metadata
from machine_learning.size_constrained_clustering import DeterministicAnnealing
# from machine_learning.size_constrained_clustering_updated import DeterministicAnnealing

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


def cluster_journal_metrics(cs_outputs_enriched_metadata, random_state=42):
    """
    Apply Deterministic Annealing clustering to journal metrics data.

    Args:
        cs_outputs_enriched_metadata (DataFrame): DataFrame containing journal metrics
        random_state (int): Random seed for reproducibility

    Returns:
        DataFrame: Original DataFrame with an additional 'cluster' column
    """
    # Make a copy of the input dataframe - avoid modifying the original
    df = cs_outputs_enriched_metadata.copy()

    # Explicitly list-out the numeric columns (needed for clustering)
    features = ['SNIP', 'SJR', 'Cite_Score', 'total_citations']

    # Handle missing values: Replace using median, this avoids: 'RuntimeWarning: invalid value encountered in divide'
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].median())

    # Extract the features for clustering
    X = df[features].values

    # Standardize the features - ensures all features contribute equally
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Configure cluster
    n_clusters = 3                  # Set number of clusters = 3
    distribution = [0.3, 0.5, 0.2]  # 30%, 50%, 20% => Expected distribution of data points

    # Initialize and fit the clustering model
    model = DeterministicAnnealing(
        n_clusters=n_clusters,
        distribution=distribution,
        max_iters=1000,
        distance_func=cdist,
        np_seed=random_state,
        T=None
    )

    model.fit(X_scaled, enforce_cluster_distribution=True)

    # Get cluster labels
    labels = model.labels_

    # Add column to dataframe that represents which cluster (cluster label) the record (data point) is assigned.
    df['cluster'] = labels

    # Verify cluster distribution - Prints the actual distribution of data points
    cluster_counts = df['cluster'].value_counts(normalize=True).sort_index()
    print("Cluster distribution:")
    for i, percentage in enumerate(cluster_counts):
        print(f"Cluster {i}: {percentage:.1%}")

    return df





cs_outputs_enriched_metadata = create_cs_outputs_enriched_metadata() # todo, just load the parquet file instead
cs_outputs_enriched_metadata = cs_outputs_enriched_metadata[['SNIP', 'SJR', 'Cite_Score', 'total_citations']]
cs_outputs_enriched_metadata = cs_outputs_enriched_metadata.head(1000)

clustered_df = cluster_journal_metrics(cs_outputs_enriched_metadata)

"""
Cluster distribution: Albert's implementation, and not enforcing ratios, head(1000).
Cluster 0: 15.7%
Cluster 1: 73.9%
Cluster 2: 10.4%

If ratios enforced:
Cluster 0: 30.0%
Cluster 1: 50.0%
Cluster 2: 20.0%
-------------------------------------------------------------------------------------------
Cluster distribution: Albert's implementation, and not enforcing ratios, entire DF.
Cluster 0: 37.1%
Cluster 1: 62.9%

If ratios enforced:
Cluster 0: 30.0%
Cluster 1: 50.0%
Cluster 2: 20.0%
----------------------------------------------------------------------------------------------
Cluster distribution: Updated clustering, not enforcing, head(1000):
Cluster 0: 38.8%
Cluster 1: 34.5%
Cluster 2: 26.7%
----------------------------------------------------------------------------------------------
Cluster distribution: Updated clustering, not enforcing, entire DF:
Cluster 0: 29.5%
Cluster 1: 47.2%
Cluster 2: 23.4%
"""
