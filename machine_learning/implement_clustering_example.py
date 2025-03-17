from machine_learning.create_cs_dataset import create_cs_dataset
from machine_learning.size_constrained_clustering import DeterministicAnnealing

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





cs_outputs_enriched_metadata = create_cs_dataset()
cs_outputs_enriched_metadata = cs_outputs_enriched_metadata[['SNIP', 'SJR', 'Cite_Score', 'total_citations']]
cs_outputs_enriched_metadata = cs_outputs_enriched_metadata.head(1000)

clustered_df = cluster_journal_metrics(cs_outputs_enriched_metadata)
