from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cdist

from machine_learning.cluster_performance_evaluation import get_cluster_evaluation_metrics, \
    update_total_evaluation_metrics, compute_cluster_evaluation_metrics, compute_clustering_accuracy, \
    update_total_divergence_metrics, get_divergence_metrics, compute_divergence_metrics

from machine_learning.cs_output_results import enhance_score_distribution, get_cs_output_results, \
    get_high_scoring_universities
from machine_learning.feature_engineering import get_cs_outputs_df
from machine_learning.high_low_output_comparison import analyse_clusters

from machine_learning.size_constrained_clustering import DeterministicAnnealing


def main():
    """
    Train and evaluate the clustering models using University-Based Leave-One-Out Cross-Validation
    """

    # Metrics:

    # Output Metrics:
    # normalised_citations, top_citation_percentile, field_weighted_citation_impact, field_weighted_views_impact

    # Journal Metrics:
    # SNIP, SJR, Cite_Score

    # List of metrics used to train the clustering models: (pick any from the metrics listed above)
    features = [
        'normalised_citations', 'top_citation_percentile'
    ]

    Leave_one_out_cross_validation(features)

def cluster_journal_metrics(
        train_df, predict_df, features, n_clusters, distribution, random_state=42,
        scale="Standard", handle_missing_data="Median"
):
    """
    Using training-data, train a clustering model that is constrained by size defined by the specified distribution.
    Use the trained cluster, to predict the cluster assignments of all data-points in the test-set
    Apply pre-processing like handling missing values and scaling data

    :param train_df: Data-points used to train the cluster models
    :param predict_df: Data-points used to test the cluster models
    :param features: List of features used to train the clustering models
    :param n_clusters: Number of clusters (2 - high- and low-scoring outputs)
    :param distribution: Distribution of training data for high- and low-scoring outputs
    :param random_state: Random seed
    :param scale: Scaling technique applied in pre-processing. Values are "Standard" or "Normal"
    :param handle_missing_data: Statistic for replacing missing values: "Mean", "Median", or "Mode"
    :return:
        1. train: Data-points used to train model with cluster assignments
        2. predicted: Data-points used to test model with clustering assignments
        3. cluster_evaluation_metrics: Metrics used to assess the clusters created using the training data
    """
    # Make a copy of the input dataframes
    train = train_df.copy()

    # Hashmap mapping each feature to the statistic used to replace missing values - mean, median, or mode.
    # The statistics are obtained from the training data to avoid leaking information while testing models
    imputation_values = {}

    if handle_missing_data == "Median":
        # For each feature, store its median from training data - used to replace missing values
        for feature in features:
            imputation_values[feature] = train[feature].median()
            train[feature] = train[feature].fillna(imputation_values[feature])

    elif handle_missing_data == "Mean":
        # For each feature, store its mean from training data - used to replace missing values
        for feature in features:
            imputation_values[feature] = train[feature].mean()
            train[feature] = train[feature].fillna(imputation_values[feature])

    elif handle_missing_data == "Mode":
        # For each feature, store its mode from training data - used to replace missing values
        for feature in features:
            mode_value = train[feature].mode()
            # In case of tie, mode returns multiple values - pick the first one
            imputation_values[feature] = mode_value.iloc[0] if not mode_value.empty else None
            train[feature] = train[feature].fillna(imputation_values[feature])

    # Extract features for clustering
    X_train = train[features].values

    # Standardize the features
    if scale == "Standard":
        scaler = StandardScaler()
    elif scale == "Normal":
        scaler = MinMaxScaler()

    # Scalar must be fit on the training data only, if fit on the entire dataset, it would leak information from test set

    # Fits the scalar to the training data
    # And transforms the training data by using the scalar to scale features
    X_train_scaled = scaler.fit_transform(X_train)

    # Set the number of clusters and desired distribution
    n_clusters = n_clusters # 2 clusters: high- and low-scoring outputs
    distribution = distribution # For example, if desired distribution = 40%, 60%, then distribution = [0.4, 0.6]

    # Using the imported Deterministic Annealing size-constrained clustering algorithm, obtain the model and fit the model
    # on the scaled training data i.e. use the clustering model to make predictions about which cluster each data-point in
    # the training set belongs to
    model = DeterministicAnnealing(
        n_clusters=n_clusters,
        distribution=distribution,
        max_iters=3000,
        distance_func=cdist,
        np_seed=random_state,
        T=None
    )

    model.fit(X_train_scaled)

    # Get cluster labels for training data
    train_labels = model.labels_

    # Obtain the metrics evaluating clustering performance of the clusters created using the training data
    cluster_evaluation_metrics = get_cluster_evaluation_metrics(model, X_train_scaled, train_labels)

    # Add cluster labels to the training dataframe
    train['cluster'] = train_labels

    predicted = predict_df.copy()

    # Replace the predicted (test) dataframe's missing values using statistics from the training-set (mean, mode, median)
    # Should not use statistics from the testing-set to avoid information leak to models
    for feature in features:
        predicted[feature] = predicted[feature].fillna(imputation_values[feature])

    # Extract features for prediction
    X_predict = predicted[features].values

    # Scale the testing data points using the same scalar used to fit the training data, again, avoiding leaking information
    X_predict_scaled = scaler.transform(X_predict)

    # Predict cluster labels
    predict_labels = model.predict(X_predict_scaled)

    # Add predicted cluster labels to the prediction dataframe
    predicted['cluster'] = predict_labels

    return train, predicted, cluster_evaluation_metrics

def infer_cluster_labels(cluster_training_df, cs_output_results_enhanced_df):
    """
    Algorithm to identify which cluster corresponds to high-scoring outputs and which to low-scoring ones.
    :param cluster_training_df: DataFrame of outputs used to train the clustering model with their cluster assignments
    :param cs_output_results_enhanced_df: DataFrame of CS output results
    :return: Hash-map mapping the cluster number (0 and 1) to the cluster it represents (high or low scoring outputs)
    """
    # Obtain a DataFrame of universities having only high-scoring outputs
    high_scoring_universities = get_high_scoring_universities(cs_output_results_enhanced_df)

    # Filter all CS outputs for outputs published by universities whose outputs were all rated high-scoring
    high_scoring_cs_outputs_df = cluster_training_df[
        cluster_training_df["Institution UKPRN code"].isin(high_scoring_universities["Institution code (UKPRN)"])
    ]

    # Count occurrences of the high-scoring CS outputs in each cluster
    cluster_counts = high_scoring_cs_outputs_df["cluster"].value_counts()

    # Determine which cluster has more of the high-scoring CS outputs
    data_points_in_cluster_0 = cluster_counts.get(0, 0)
    data_points_in_cluster_1 = cluster_counts.get(1, 0)

    # The cluster which has a greater amount of the high-scoring CS outputs is said to represent the high-scoring outptus
    # while the other cluster represents the low-scoring outputs
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
    """
    Compute the percentages of actual high- and low-scoring outputs of a university
    :param high_scoring_output_count: Number of outputs of a university that were scored high
    :param low_scoring_output_count: Number of outputs of a university that were scored low
    :return: Percentages of the high- and low-scoring outputs of a university
    """
    total_output_count = high_scoring_output_count + low_scoring_output_count

    high_scoring_output_percentage = (high_scoring_output_count / total_output_count) * 100
    low_scoring_output_percentage = (low_scoring_output_count / total_output_count) * 100

    return (
        high_scoring_output_percentage,
        low_scoring_output_percentage
    )
    # Alternatively, [4*+3*, 2*+1*+unclassified] can be returned
    # Currently return values are precise
    # For example, currently 10007800 (University of the West of Scotland), returns (69.84126984126983, 30.158730158730158)
    # Adding the results ratios [6.3, 63.5, 28.6, 1.6, 0] would return (69.8, 30.2)

def log_training_data_cluster_distribution(train, cluster_label_mapping, distribution):
    """
    Log the distribution of high and low scoring outputs for the data used to train the cluster
    :param train: DataFrame containing data points used to train the cluster
    :param cluster_label_mapping: Hashmap mapping each cluster (0, 1) to a label (high or low scoring)
    :param distribution: Distribution of high and low scoring outputs
    """
    cluster_counts = train['cluster'].value_counts(normalize=True)
    print("Training cluster distribution:")
    print(f"Provided distribution = {distribution}")
    for i, percentage in enumerate(cluster_counts):
        print(f"Cluster {cluster_label_mapping[i]}: {percentage:.1%}")

def log_testing_data_cluster_distribution(predicted, cluster_label_mapping):
    """
    Log the distribution of high and low scoring outputs for the data used to train the cluster
    :param predicted: DataFrame containing data points used to test the cluster (fold)
    :param cluster_label_mapping: Hashmap mapping each cluster (0, 1) to a label (high or low scoring)
    """
    pred_cluster_counts = predicted['cluster'].value_counts(normalize=True)

    print(f"[debug] pred_cluster_counts: {pred_cluster_counts}")
    print("\nPrediction cluster distribution:")
    for i, ratio in enumerate(pred_cluster_counts):
        print(f"Cluster {cluster_label_mapping[i]}: {ratio:.1%}")

def get_predicted_output_score_percentages(predicted, cluster_label_mapping):
    """
    Compute the percentages of high- and low-scoring outputs as predicted by the clustering model (when model is tested)
    :param predicted: DataFrame containing data points used to test the cluster (fold)
    :param cluster_label_mapping: Hashmap mapping each cluster (0, 1) to a label (high or low scoring)
    :return: The percentages of high- and low-scoring outputs as predicted by the clustering model
    """
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
def Leave_one_out_cross_validation(cluster_features):
    """
    Train and evaluate the size-constrained cluster models, passing in a list of features to train on
    :param cluster_features: List of features used to train the clustering models
    """
    # Percentages of outputs of the current university (fold / test-set) that are high-scoring:
    actual_high_scoring_output_percentages = []
    predicted_high_scoring_output_percentages = []

    # Percentages of outputs of the current university (fold / test-set) that are low-scoring:
    actual_low_scoring_output_percentages = []
    predicted_low_scoring_output_percentages = []

    # Obtained the feature-engineered DataFrame of enhanced CS outputs metrics
    cs_outputs_enriched_metadata = get_cs_outputs_df(cluster_features)

    # Obtain the DataFrame of enhanced REF CS Output Quality results containing number of high- and low-scoring outputs
    # for each university
    cs_output_results_df = get_cs_output_results()
    cs_output_results_enhanced_df = enhance_score_distribution(cs_output_results_df, cs_outputs_enriched_metadata)

    # Obtain the total count of high- and low-scoring outputs across all universities
    total_high_scoring_output_count = cs_output_results_enhanced_df['high_scoring_outputs'].sum()
    total_low_scoring_output_count = cs_output_results_enhanced_df['low_scoring_outputs'].sum()

    # Cluster Evaluation Metrics:
    total_folds = 0
    # Internal cluster indices to assess cluster quality
    total_evaluation_metrics = {
        "total_silhouette_score": 0,
        "total_davies_bouldin_score": 0,
        "total_calinski_harabasz_score": 0,
        "total_inertia": 0,
        "total_bcss": 0
    }
    # Divergence metrics to assess cluster accuracy
    total_divergence_metrics = {
        "total_kl_divergence": 0,
        "total_js_divergence": 0,
        "total_tvd": 0
    }

    # University-Based Leave-One-Out Cross-Validation for Clustering
    # Use the outputs from oen university at a time as the testing set (fold) and all other outputs as the training set
    for ukprn in cs_output_results_enhanced_df['Institution code (UKPRN)']:

        # The current university will be used to test the cluster created by training on all other university metadata
        is_curr_university_result = cs_output_results_enhanced_df['Institution code (UKPRN)'] == ukprn
        # Obtain the REF CS output quality results of the testing university
        curr_university_cs_output_result_df = cs_output_results_enhanced_df[is_curr_university_result]

        # Obtain the test university's counts of high- and low-scoring outputs
        curr_uni_high_scoring_output_count = curr_university_cs_output_result_df['high_scoring_outputs'].item()
        curr_uni_low_scoring_output_count = curr_university_cs_output_result_df['low_scoring_outputs'].item()

        # Using counts, compute the actual percentages of high- and low-scoring outputs for the current (test) university
        actual_high_scoring_output_percentage, actual_low_scoring_output_percentage = get_actual_output_score_percentages(
            curr_uni_high_scoring_output_count,
            curr_uni_low_scoring_output_count
        )
        actual_high_scoring_output_percentages.append(actual_high_scoring_output_percentage)
        actual_low_scoring_output_percentages.append(actual_low_scoring_output_percentage)

        is_curr_university_output = cs_outputs_enriched_metadata['Institution UKPRN code'] == ukprn
        # Use the CS outputs from all universities (excluding current) to create the two clusters i.e. train the model
        training_outputs_df = cs_outputs_enriched_metadata[~is_curr_university_output]
        # Use current university's CS outputs to evaluate the effectiveness of clustering
        testing_output_df = cs_outputs_enriched_metadata[is_curr_university_output]

        # Obtain the number of high- and low-scoring outputs in the training dataset
        # Add the counts of the high (or low) scoring outputs from all universities but the one used for testing
        high_scoring_cluster_output_count = total_high_scoring_output_count - curr_uni_high_scoring_output_count
        low_scoring_cluster_output_count = total_low_scoring_output_count - curr_uni_low_scoring_output_count

        # Using counts, compute the target distribution of high- and low-scoring outputs in the clusters
        # This defines the size constrains for the DA clustering
        cluster_output_count = high_scoring_cluster_output_count + low_scoring_cluster_output_count

        high_scoring_output_cluster_distribution =  (high_scoring_cluster_output_count / cluster_output_count)
        low_scoring_output_cluster_distribution = (low_scoring_cluster_output_count / cluster_output_count)

        cluster_distribution = [high_scoring_output_cluster_distribution, low_scoring_output_cluster_distribution]

        # Using the DA clustering algorithm, build the model using the training set, and on the test set, make
        # cluster assignments for all data-points. Also obtain the evaluation metrics of the cluster created

        train, predicted, cluster_evaluation_metrics = cluster_journal_metrics(
            training_outputs_df, # All data points (outputs) excluding ones belonging to current university
            testing_output_df,   # Data points (outputs) of current university (fold / test-set)
            features=cluster_features,   # Features using which clusters are created
            n_clusters=2,        # Clusters: High scoring outputs & Low scoring outputs
            distribution=cluster_distribution, # Target distribution of the training data's data points across 2 clusters
            scale = "Standard", # Feature Scaling Technique: "Standard" or "Normal"
            handle_missing_data = "Median" # Statistic for replacing missing values: "Mean", "Median", or "Mode"
        )

        # Update the cluster evaluation metrics using the cluster obtained from the DA clustering algorithm
        total_evaluation_metrics = update_total_evaluation_metrics(total_evaluation_metrics, cluster_evaluation_metrics)

        # Infer the labels of clusters to something more meaningful than 0 and 1
        # Returns a dictionary mapping each cluster to the output type (high/low scoring) it represents
        cluster_label_mapping = infer_cluster_labels(train, cs_output_results_enhanced_df)

        # Verify cluster distribution for training data
        # log_training_data_cluster_distribution(train, cluster_label_mapping, cluster_distribution)

        # Show prediction cluster distribution
        # log_testing_data_cluster_distribution(predicted, cluster_label_mapping)

        # For the test-set which now has data-points assigned to a cluster, compute the distribution of predicted
        # high- and low-scoring outputs
        (
            predicted_high_scoring_output_percentage,
            predicted_low_scoring_output_percentage
        ) = get_predicted_output_score_percentages(predicted, cluster_label_mapping)

        predicted_high_scoring_output_percentages.append(predicted_high_scoring_output_percentage)
        predicted_low_scoring_output_percentages.append(predicted_low_scoring_output_percentage)

        predicted_distribution = [predicted_high_scoring_output_percentage/100, predicted_low_scoring_output_percentage/100]
        actual_distribution = [actual_high_scoring_output_percentage/100, actual_low_scoring_output_percentage/100]

        # Using the predicted and actual distribution of high- and low-scoring outputs, compute divergence metrics
        # to quantify close closely the two distributions match
        divergence_metrics = get_divergence_metrics(predicted_distribution, actual_distribution)
        total_divergence_metrics = update_total_divergence_metrics(total_divergence_metrics, divergence_metrics)

        total_folds += 1

        # Analyse the trained clusters to identify which features are strong indicators of clustering quality
        if ukprn == 10007833:
            # Instead of analysing of every training set (90 in total), only do it once when the testing set is Wrexham Uni
            # This is because this university has the least number of outputs (9) meaning this fold results in the highest
            # number of outputs in the training set compared to all other folds
            analyse_clusters(train, cluster_label_mapping)

    print(f"Features Used to Train Model: {cluster_features}\n")

    # After cross-validation where every university was the test-set (fold) once, and the cluster evaluation metrics were
    # computed for every test-set, compute the cluster metrics evaluation metrics - average across all folds:

    # Compute regression metrics to assess cluster accuracy
    compute_clustering_accuracy(
        actual_high_scoring_output_percentages,
        predicted_high_scoring_output_percentages,
        actual_low_scoring_output_percentages,
        predicted_low_scoring_output_percentages
    )

    # Compute divergence metrics to assess cluster accuracy
    compute_divergence_metrics(
        total_divergence_metrics,
        total_folds
    )

    # Compute internal cluster indices to assess cluster quality
    compute_cluster_evaluation_metrics(
        total_evaluation_metrics,
        total_folds
    )



if __name__ == "__main__":
    main()
