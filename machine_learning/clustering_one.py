from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist

from machine_learning.cluster_performance_evaluation import get_cluster_evaluation_metrics, \
    update_total_evaluation_metrics, compute_cluster_evaluation_metrics, compute_clustering_accuracy, \
    update_total_divergence_metrics, get_divergence_metrics, compute_divergence_metrics

from machine_learning.cs_output_results import enhance_score_distribution, get_cs_output_results, \
    get_high_scoring_universities
from machine_learning.feature_engineering import get_cs_outputs_df, get_cluster_features

# from machine_learning.size_constrained_clustering import DeterministicAnnealing
from machine_learning.size_constrained_clustering_updated import DeterministicAnnealing

def main():
    # or maybe just represent using a number like 1,2,3.. and create a table in report map input-set
    input_set = {'citations', 'journal metrics', 'scival output metrics'}
    Leave_one_out_cross_validation(input_set)

def cluster_journal_metrics(
        train_df, predict_df, features, n_clusters, distribution, random_state=42,
        scale="Standard", handle_missing_data="Median"
):
    # Make a copy of the input dataframes
    train = train_df.copy()

    # Select the numeric columns for clustering
    # features = ['SNIP', 'SJR', 'Cite_Score', 'total_citations']

    if handle_missing_data == "Median":
        # Store medians from training data for consistent imputation
        medians = {}
        for feature in features:
            medians[feature] = train[feature].median()
            train[feature] = train[feature].fillna(medians[feature])

    # Extract features for clustering
    X_train = train[features].values

    # Standardize the features
    if scale == "Standard":
        scaler = StandardScaler()
    elif scale == "Normal":
        scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train) # X_train_scaled = X_train

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

    cluster_evaluation_metrics = get_cluster_evaluation_metrics(model, X_train_scaled, train_labels)

    # Add cluster labels to the training dataframe
    train['cluster'] = train_labels

    predicted = predict_df.copy()

    if handle_missing_data == "Median":
        # Apply the same preprocessing as training data
        for feature in features:
            predicted[feature] = predicted[feature].fillna(medians[feature])

    # Extract features for prediction
    X_predict = predicted[features].values

    # Standardise (scale) the testing data points using the same scaler used to fit the training data
    X_predict_scaled = scaler.transform(X_predict) # X_predict_scaled = X_predict

    # Predict cluster labels
    predict_labels = model.predict(X_predict_scaled)

    # Add predicted cluster labels to the prediction dataframe
    predicted['cluster'] = predict_labels

    return train, predicted, cluster_evaluation_metrics

def infer_cluster_labels(cluster_training_df, cs_output_results_enhanced_df):
    high_scoring_universities = get_high_scoring_universities(cs_output_results_enhanced_df)

    # Filter cluster_training_df for universities in high_scoring_universities => high scoring unis datapoints
    high_scoring_cs_outputs_df = cluster_training_df[
        cluster_training_df["Institution UKPRN code"].isin(high_scoring_universities["Institution code (UKPRN)"])
    ]

    # Count occurrences of each cluster
    cluster_counts = high_scoring_cs_outputs_df["cluster"].value_counts()

    # Determine which cluster has more datapoints
    data_points_in_cluster_0 = cluster_counts.get(0, 0)
    data_points_in_cluster_1 = cluster_counts.get(1, 0)
    # print(data_points_in_cluster_0, data_points_in_cluster_1)

    if data_points_in_cluster_0 > data_points_in_cluster_1:
        return {
            0: 'high_scoring_outputs',
            1: 'low_scoring_outputs'
        }
    else:
        return {
            0: 'low_scoring_outputs',
            1: 'high_scoring_outputs'
        }


def get_actual_output_score_percentages(high_scoring_output_count, low_scoring_output_count):
    total_output_count = high_scoring_output_count + low_scoring_output_count

    high_scoring_output_percentage = (high_scoring_output_count / total_output_count) * 100
    low_scoring_output_percentage = (low_scoring_output_count / total_output_count) * 100

    return (
        high_scoring_output_percentage,
        low_scoring_output_percentage
    )
    # Alternatively you can just return [4*+3*, 2*+1*+unclassified]
    # Currently returns it super-precise, but using 4*,3*... from results gives rounded ratios
    # For example, 10007800 (University of the West of Scotland), I return (69.84126984126983, 30.158730158730158)
    # but just adding the results ratios [6.3, 63.5, 28.6, 1.6, 0] would return (69.8, 30.2)

def log_training_data_cluster_distribution(train, cluster_label_mapping, distribution):
    cluster_counts = train['cluster'].value_counts(normalize=True)
    print("Training cluster distribution:")
    print(f"Provided distribution = {distribution}")
    for i, percentage in enumerate(cluster_counts):
        # print(f"Cluster {i}: {percentage:.1%}")
        print(f"Cluster {cluster_label_mapping[i]}: {percentage:.1%}")

def log_testing_data_cluster_distribution(predicted, cluster_label_mapping):
    pred_cluster_counts = predicted['cluster'].value_counts(normalize=True)

    print(f"[debug] pred_cluster_counts: {pred_cluster_counts}")
    print("\nPrediction cluster distribution:")
    for i, ratio in enumerate(pred_cluster_counts):
        # print(f"Cluster {i}: {ratio:.1%}")
        print(f"Cluster {cluster_label_mapping[i]}: {ratio:.1%}")

def get_predicted_output_score_percentages(predicted, cluster_label_mapping):
    predicted_cluster_distribution_high_scoring_outputs = 0
    predicted_cluster_distribution_low_scoring_outputs = 0

    pred_cluster_counts = predicted['cluster'].value_counts(normalize=True) # normalise to convert counts to proportions

    for i, ratio in enumerate(pred_cluster_counts):
        percentage = ratio * 100
        if cluster_label_mapping[i] == "high_scoring_outputs":
            predicted_cluster_distribution_high_scoring_outputs = percentage
        elif cluster_label_mapping[i] == "low_scoring_outputs":
            predicted_cluster_distribution_low_scoring_outputs = percentage

    return (
        predicted_cluster_distribution_high_scoring_outputs,
        predicted_cluster_distribution_low_scoring_outputs
    )

# Leave-one-out cross-validation - creates a total of 90 models
def Leave_one_out_cross_validation(input_set):
    actual_high_scoring_output_percentages = []
    predicted_high_scoring_output_percentages = []

    actual_low_scoring_output_percentages = []
    predicted_low_scoring_output_percentages = []

    cs_outputs_enriched_metadata = get_cs_outputs_df(input_set)
    cluster_features = get_cluster_features(input_set)

    cs_output_results_df = get_cs_output_results()
    cs_output_results_enhanced_df = enhance_score_distribution(cs_output_results_df, cs_outputs_enriched_metadata)

    total_high_scoring_output_count = cs_output_results_enhanced_df['high_scoring_outputs'].sum()
    total_low_scoring_output_count = cs_output_results_enhanced_df['low_scoring_outputs'].sum()

    # Cluster Evaluation Metrics:
    total_folds = 0
    total_evaluation_metrics = {
        "total_silhouette_score": 0,
        "total_davies_bouldin_score": 0,
        "total_calinski_harabasz_score": 0,
        "total_inertia": 0,
        "total_bcss": 0
    }
    total_divergence_metrics = {
        "total_kl_divergence": 0,
        "total_js_divergence": 0,
        "total_tvd": 0
    }

    for ukprn in cs_output_results_enhanced_df['Institution code (UKPRN)']:

        # The current university will be used to test the cluster created by training on  all other university metadata
        is_curr_university_result = cs_output_results_enhanced_df['Institution code (UKPRN)'] == ukprn
        curr_university_cs_output_result_df = cs_output_results_enhanced_df[is_curr_university_result]

        curr_uni_high_scoring_output_count = curr_university_cs_output_result_df['high_scoring_outputs'].item() # Expected
        curr_uni_low_scoring_output_count = curr_university_cs_output_result_df['low_scoring_outputs'].item() # Expected

        # Get the actual percentages of high and low scoring outputs for the current university
        actual_high_scoring_output_percentage, actual_low_scoring_output_percentage = get_actual_output_score_percentages(
            curr_uni_high_scoring_output_count,
            curr_uni_low_scoring_output_count
        )
        actual_high_scoring_output_percentages.append(actual_high_scoring_output_percentage)
        actual_low_scoring_output_percentages.append(actual_low_scoring_output_percentage)

        is_curr_university_output = cs_outputs_enriched_metadata['Institution UKPRN code'] == ukprn
        # Use the CS outputs from all universities (excluding current) to create the two clusters
        training_outputs_df = cs_outputs_enriched_metadata[~is_curr_university_output]
        # Use current university's CS outputs to evaluate the effectiveness of clustering
        testing_output_df = cs_outputs_enriched_metadata[is_curr_university_output]

        high_scoring_cluster_output_count = total_high_scoring_output_count - curr_uni_high_scoring_output_count
        low_scoring_cluster_output_count = total_low_scoring_output_count - curr_uni_low_scoring_output_count

        cluster_output_count = high_scoring_cluster_output_count + low_scoring_cluster_output_count

        high_scoring_output_cluster_distribution =  (high_scoring_cluster_output_count / cluster_output_count)
        low_scoring_output_cluster_distribution = (low_scoring_cluster_output_count / cluster_output_count)

        cluster_distribution = [high_scoring_output_cluster_distribution, low_scoring_output_cluster_distribution]

        train, predicted, cluster_evaluation_metrics = cluster_journal_metrics(
            training_outputs_df, # All data points (outputs) excluding ones belonging to current university
            testing_output_df,   # Data points (outputs) of current university
            features=cluster_features,   # Features using which clusters are created
            n_clusters=2,        # Clusters: High scoring outputs & Low scoring outputs
            distribution=cluster_distribution,
            scale = "Standard",
            handle_missing_data = "Median"
        )

        total_evaluation_metrics = update_total_evaluation_metrics(total_evaluation_metrics, cluster_evaluation_metrics)

        # Returns possibly a dictionary indicating which cluster maps to high_scoring or low_scoring
        cluster_label_mapping = infer_cluster_labels(train, cs_output_results_enhanced_df)

        # Verify cluster distribution for training data
        # log_training_data_cluster_distribution(train, cluster_label_mapping, cluster_distribution)

        # Show prediction cluster distribution
        # log_testing_data_cluster_distribution(predicted, cluster_label_mapping)

        (
            predicted_high_scoring_output_percentage,
            predicted_low_scoring_output_percentage
        ) = get_predicted_output_score_percentages(predicted, cluster_label_mapping)

        predicted_high_scoring_output_percentages.append(predicted_high_scoring_output_percentage)
        predicted_low_scoring_output_percentages.append(predicted_low_scoring_output_percentage)

        predicted_distribution = [predicted_high_scoring_output_percentage/100, predicted_low_scoring_output_percentage/100]
        actual_distribution = [actual_high_scoring_output_percentage/100, actual_low_scoring_output_percentage/100]

        divergence_metrics = get_divergence_metrics(predicted_distribution, actual_distribution)
        total_divergence_metrics = update_total_divergence_metrics(total_divergence_metrics, divergence_metrics)

        total_folds += 1

    compute_clustering_accuracy(
        actual_high_scoring_output_percentages,
        predicted_high_scoring_output_percentages,
        actual_low_scoring_output_percentages,
        predicted_low_scoring_output_percentages
    )

    compute_divergence_metrics(
        total_divergence_metrics,
        total_folds
    )

    compute_cluster_evaluation_metrics(
        total_evaluation_metrics,
        total_folds
    )


# In the cs_output_results_df dataframe, create a new field called total submissions
# Maybe even in the process_cs_output_results() function
# Calculate the value of this field, based on the cs_outputs_enriched_metadata df.

# Now total number of 4* ->         (total 4* - 4* of aber uni) / total 4*
# More efficient compared to ->     (uni1 4* + uni2 4* + ...... ) / total 4*

# Possible just implement the k-means [1 off k-means with 90 models.]

# clustered_df = cluster_journal_metrics(cs_outputs_enriched_metadata, 3, [0.3, 0.5, 0.2])


if __name__ == "__main__":
    main()


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

# I think make k validation a function, and then get the DFs and then make functions within it.
# This way you dont have to pass in arguments like cs_outputs_enriched_metadata around, since they're in scope.
