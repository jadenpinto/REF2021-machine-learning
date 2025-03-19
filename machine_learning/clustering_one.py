import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from machine_learning.create_cs_outputs_enriched_metadata import create_cs_outputs_enriched_metadata
from machine_learning.cs_output_results import get_cs_outputs_enriched_metadata, enhance_score_distribution, \
    get_cs_output_results

# from machine_learning.size_constrained_clustering import DeterministicAnnealing
from machine_learning.size_constrained_clustering_updated import DeterministicAnnealing


def cluster_journal_metrics(train_df, predict_df, n_clusters, distribution, random_state=42):
    """
    Apply Deterministic Annealing clustering to journal metrics data.
    Train on train_df and optionally predict on predict_df.

    Args:
        train_df (DataFrame): DataFrame containing journal metrics for training
        predict_df (DataFrame, optional): DataFrame containing journal metrics for prediction
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (trained_df, predicted_df) - DataFrames with cluster columns
               If predict_df is None, predicted_df will be None
    """
    # Make a copy of the input dataframes
    train = train_df.copy()

    # Select the numeric columns for clustering
    features = ['SNIP', 'SJR', 'Cite_Score', 'total_citations']

    # Store medians from training data for consistent imputation
    medians = {}
    for feature in features:
        medians[feature] = train[feature].median()
        train[feature] = train[feature].fillna(medians[feature])

    # Extract features for clustering
    X_train = train[features].values

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Set the number of clusters and desired distribution
    n_clusters = n_clusters
    distribution = distribution # [0.3, 0.5, 0.2]  # 30%, 50%, 20%

    model = DeterministicAnnealing(
        n_clusters=n_clusters,
        distribution=distribution,
        max_iters=1000,
        distance_func=cdist,
        np_seed=random_state,
        T=None
    )

    model.fit(X_train_scaled) #, enforce_cluster_distribution=True)

    # Get cluster labels for training data
    train_labels = model.labels_

    # Add cluster labels to the training dataframe
    train['cluster'] = train_labels

    # Verify cluster distribution for training data
    cluster_counts = train['cluster'].value_counts(normalize=True).sort_index()
    print("Training cluster distribution:")
    for i, percentage in enumerate(cluster_counts):
        print(f"Cluster {i}: {percentage:.1%}")

    # Handle prediction dataframe if provided
    predicted = None
    if predict_df is not None:
        predicted = predict_df.copy()

        # Apply the same preprocessing as training data
        for feature in features:
            predicted[feature] = predicted[feature].fillna(medians[feature])

        # Extract features for prediction
        X_predict = predicted[features].values

        # Standardize using the same scaler as training data
        X_predict_scaled = scaler.transform(X_predict)

        # Predict cluster labels
        predict_labels = model.predict(X_predict_scaled)

        # Add predicted cluster labels to the prediction dataframe
        predicted['cluster'] = predict_labels

        # Show prediction cluster distribution
        pred_cluster_counts = predicted['cluster'].value_counts(normalize=True).sort_index()
        print("\nPrediction cluster distribution:")
        for i, percentage in enumerate(pred_cluster_counts):
            print(f"Cluster {i}: {percentage:.1%}")

    return train, predicted



def get_cs_outputs_by_university(cs_outputs_enriched_metadata):
    """
    Returns a hashmap mapping Institution UKPRN code to the count of records
    in the dataframe.

    Parameters:
    cs_outputs_enriched_metadata (pd.DataFrame): DataFrame containing 'Institution UKPRN code'.

    Returns:
    dict: A dictionary mapping each Institution UKPRN code to its count.

    Note:
    # cs_outputs_grouped_by_uni_counts = get_cs_outputs_by_university(cs_outputs_enriched_metadata)
    # Don't think I need this.
    # actually might need it.
    # alt: is to lookup Institution code (UKPRN) in the enhanced results dataframe and get total_university_outputs
    # actually don't think we need it - cause you're iterating row by row through it so no need for lookup
    # Get the global high and low, and you can just subtract for training
    # testing is current high and low (as is)
    """
    return cs_outputs_enriched_metadata['Institution UKPRN code'].value_counts().to_dict()


cs_output_results_df = get_cs_output_results()
cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()

cs_output_results_enhanced_df = enhance_score_distribution(cs_output_results_df, cs_outputs_enriched_metadata)

total_high_scoring_outputs = cs_output_results_enhanced_df['high_scoring_outputs'].sum()
total_low_scoring_outputs = cs_output_results_enhanced_df['low_scoring_outputs'].sum()

# Leave-one-out cross-validation - creates a total of 90 models

for ukprn in cs_output_results_enhanced_df['Institution code (UKPRN)']:

    if ukprn == 10007856:                                             # Remove this later, this is just for debugging.

        curr_uni_result = cs_output_results_enhanced_df['Institution code (UKPRN)'] == ukprn
        training_cs_output_results_df = cs_output_results_enhanced_df[~curr_uni_result]
        testing_cs_output_results_df = cs_output_results_enhanced_df[curr_uni_result]

        curr_uni_high_scoring_outputs = testing_cs_output_results_df['high_scoring_outputs'].item() # Expected
        curr_uni_low_scoring_outputs = testing_cs_output_results_df['low_scoring_outputs'].item() # Expected

        curr_uni_outputs = cs_outputs_enriched_metadata['Institution UKPRN code'] == ukprn
        training_outputs_df = cs_outputs_enriched_metadata[~curr_uni_outputs]
        testing_output_df = cs_outputs_enriched_metadata[curr_uni_outputs]

        high_scoring_cluster_outputs = total_high_scoring_outputs - curr_uni_high_scoring_outputs
        low_scoring_cluster_outputs = total_low_scoring_outputs - curr_uni_low_scoring_outputs

        total_cluster_outputs = high_scoring_cluster_outputs + low_scoring_cluster_outputs

        high_scoring_output_cluster_distribution =  (high_scoring_cluster_outputs / total_cluster_outputs)
        low_scoring_output_cluster_distribution = (low_scoring_cluster_outputs / total_cluster_outputs)


        trained_df, predicted_df = cluster_journal_metrics(
            training_outputs_df,
            testing_output_df,
            n_clusters=2,
            distribution=[high_scoring_output_cluster_distribution, low_scoring_output_cluster_distribution]
        )

        total_curr_uni_outputs = curr_uni_high_scoring_outputs + curr_uni_low_scoring_outputs
        exp_high_percent = (curr_uni_high_scoring_outputs / total_curr_uni_outputs) * 100
        exp_low_percent = (curr_uni_low_scoring_outputs / total_curr_uni_outputs) * 100

        print(exp_high_percent, exp_low_percent)


# In the cs_output_results_df dataframe, create a new field called total submissions
# Maybe even in the process_cs_output_results() function
# Calculate the value of this field, based on the cs_outputs_enriched_metadata df.

# Now total number of 4* ->         (total 4* - 4* of aber uni) / total 4*
# More efficient compared to ->     (uni1 4* + uni2 4* + ...... ) / total 4*

# Possible just implement the k-means [1 off k-means with 90 models.]

# clustered_df = cluster_journal_metrics(cs_outputs_enriched_metadata, 3, [0.3, 0.5, 0.2])


"""
Files:
1) Outputs
2) Results - get results, and get counts. todo.
3) Feature Engineering
4) Clustering

Also, you need only 2 universities which for you know are either high/low scoring
say uni A and uni B
When you training includes uni A -> use uni A to determine which cluster is low and which is high
When you training excludes uni A (i.e. you're testing for uni A) -> use uni B to determine cluster labels.

Preferable pick A and B such that they have many outputs [and an odd number]
To determine:
    assume A=all high scoring. Check counts data points of A in cluster 0 and 1.
    if majority in cluster 0, then 0 is high scoring, else 1 is high scoring.
"""


"""
curr_uni_high_scoring_outputs = testing_cs_output_results_df['high_scoring_outputs'].item()
        curr_uni_low_scoring_outputs = testing_cs_output_results_df['low_scoring_outputs'].item()
        # Expected high and low count ^
        # Predictions for clusters should match this count as close as possible.
        # if comparing ratios, might be redundant to use both high and low. For ex: if low=40, inferred tht high=60
"""
