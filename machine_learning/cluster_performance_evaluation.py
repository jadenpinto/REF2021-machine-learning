import numpy as np
from math import log2, sqrt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, r2_score

# Internal indices - No external information, evaluate

def get_cluster_evaluation_metrics(model, training_feature_array, predicted_training_labels):
    """
    Compute internal indices used to assess the quality of clusters

    :param model: Clustering model that is fitted i.e. the collection of datapoints use to train the clustering model
    :param training_feature_array: Array of features used to train the model
    :param predicted_training_labels: The labels i.e. cluster assignments of the datapoints used to train the model
    :return:
    """
    evaluation_metrics = {}

    evaluation_metrics["silhouette_score"] = silhouette_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["davies_bouldin_score"] = davies_bouldin_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["calinski_harabasz_score"] = calinski_harabasz_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["inertia"] = model.inertia_ # Within-Cluster Sum of Squares (Inertia)
    evaluation_metrics["bcss"] = model.compute_bcss(training_feature_array)  # Between-Cluster Sum of Squares

    return evaluation_metrics

def update_total_evaluation_metrics(total_evaluation_metrics, cluster_evaluation_metrics):
    """
    Update the hash-map storing the sum of all internal clustering indices of all runs so far by adding the values
    of the internal indices from the current run.

    :param total_evaluation_metrics: Hash-map containing the sum of all internal clustering indices of runs so far
    :param cluster_evaluation_metrics: Hash-map containing the internal clustering indices of the current run (fold)
    :return: Updated hash-map containing the sum of all internal clustering indices (with current run's indices added)
    """
    total_evaluation_metrics["total_silhouette_score"] += cluster_evaluation_metrics["silhouette_score"]
    total_evaluation_metrics["total_davies_bouldin_score"] += cluster_evaluation_metrics["davies_bouldin_score"]
    total_evaluation_metrics["total_calinski_harabasz_score"] += cluster_evaluation_metrics["calinski_harabasz_score"]
    total_evaluation_metrics["total_inertia"] += cluster_evaluation_metrics["inertia"]
    total_evaluation_metrics["total_bcss"] += cluster_evaluation_metrics["bcss"]

    return total_evaluation_metrics

def compute_cluster_evaluation_metrics(total_evaluation_metrics, total_folds):
    """
    Compute the final internal cluster indices by averaging the indices over all folds

    :param total_evaluation_metrics: Hash-map containing the sum of all internal clustering indices of runs so far
    :param total_folds: Total number of folds (runs) where a clustering model was built and evaluated
    """
    average_silhouette_score = total_evaluation_metrics["total_silhouette_score"] / total_folds
    average_davies_bouldin_score = total_evaluation_metrics["total_davies_bouldin_score"] / total_folds
    average_calinski_harabasz_score = total_evaluation_metrics["total_calinski_harabasz_score"] / total_folds
    average_inertia = total_evaluation_metrics["total_inertia"] / total_folds
    average_bcss = total_evaluation_metrics["total_bcss"] / total_folds

    print("Internal indices - quantify effectiveness of clustering structure")
    print(f"Average Silhouette Score: {average_silhouette_score:.4f}")
    print(f"Average Davies Bouldin Score: {average_davies_bouldin_score:.4f}")
    print(f"Average Calinski Harabasz Score: {average_calinski_harabasz_score:.4f}")
    print(f"Average Within-Cluster Sum of Squares (Inertia): {average_inertia:.4f}")
    print(f"Average Between-Cluster Sum of Squares: {average_bcss:.4f}")
    print()

# Regression metrics

def compute_clustering_accuracy(
        actual_high_scoring_output_percentages,
        predicted_high_scoring_output_percentages,
        actual_low_scoring_output_percentage,
        predicted_low_scoring_output_percentage
    ):
    """
    Compute regression metrics by comparing the actual again the predicted number of high scoring outputs of the university
    who's outputs are used to test the model (training set)

    :param actual_high_scoring_output_percentages: Array of the actual number of high scoring outputs of the university used as test-set
    :param predicted_high_scoring_output_percentages: Array of the predicted number of high scoring outputs of the university used as test-set
    :param actual_low_scoring_output_percentage: Array of the actual number of low scoring outputs of the university used as test-set
    :param predicted_low_scoring_output_percentage: Array of the predicted number of low scoring outputs of the university used as test-set
    """

    actual_high = np.array(actual_high_scoring_output_percentages)
    predicted_high = np.array(predicted_high_scoring_output_percentages)

    # Mean Absolute Error
    mae = np.mean(np.abs(actual_high - predicted_high))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual_high - predicted_high) ** 2))

    # Mean Absolute Percentage Error
    # In cases where the actual number of high scoring outputs is zero, compare using low scoring outputs instead
    mape_values = []

    predictions = len(predicted_high_scoring_output_percentages)
    for prediction in range(predictions):
        # To avoid dividing by 0, if the actual high scoring percentage is 0, use the low scoring percentages instead.
        if actual_high_scoring_output_percentages[prediction] == 0:
            mape_values.append(
                abs((actual_low_scoring_output_percentage[prediction] - predicted_low_scoring_output_percentage[prediction]) / actual_low_scoring_output_percentage[prediction]) * 100
            )
        else:
            mape_values.append(
                abs((actual_high_scoring_output_percentages[prediction] - predicted_high_scoring_output_percentages[prediction]) / actual_high_scoring_output_percentages[prediction]) * 100
            )

    mape = np.mean(mape_values)

    # R^2 Score (Coefficient of Determination) - quantify how well the predictions fit the actual values
    r2 = r2_score(actual_high_scoring_output_percentages, predicted_high_scoring_output_percentages)

    print("Regression Scores - compare percentage of predicted and actual high scoring outputs")
    # Lower is better
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")

    # [-inf, 1] => 1 means Perfect predictions.
    # 0 means no relation at all between predicted and actual values
    # Higher is better
    print(f"R^2 score: {r2:.4f}")
    print()

# Divergence metrics:

def get_divergence_metrics(predicted_distribution, actual_distribution):
    """
    Compute divergence metrics to assess the accuracy of the clustering models
    :param predicted_distribution: Predicted distribution - probability of an output being high-scoring, and low-scoring
    :param actual_distribution: Actual distribution - probability of an output being high-scoring, and low-scoring
    :return: Hash-map with the divergence metrics quantifying distance between the distributions
    """
    divergence_metrics = {}

    divergence_metrics["kl_divergence"] = kl_divergence(predicted_distribution, actual_distribution)
    divergence_metrics["js_divergence"] = js_divergence(predicted_distribution, actual_distribution)
    divergence_metrics["tvd"] = total_variation_distance(predicted_distribution, actual_distribution)

    return divergence_metrics

def update_total_divergence_metrics(total_divergence_metrics, divergence_metrics):
    """
    Update the hash-map storing the sum of all divergence of all runs so far by adding the values of the divergence metrics
    the current run (fold)

    :param total_divergence_metrics: Hash-map containing the sum of all divergence metrics of runs so far
    :param divergence_metrics: Hash-map containing the divergence metrics of the current run (fold)
    :return:  Updated hash-map containing the sum of all divergence metrics (with current run's divergence metrics added)
    """
    total_divergence_metrics["total_kl_divergence"] += divergence_metrics["kl_divergence"]
    total_divergence_metrics["total_js_divergence"] += divergence_metrics["js_divergence"]
    total_divergence_metrics["total_tvd"] += divergence_metrics["tvd"]

    return total_divergence_metrics

def compute_divergence_metrics(total_divergence_metrics, total_folds):
    """
    Compute and print the overall divergence metrics between the probability distributions of
    the actual and the predicted high- and low-scoring output counts by averaging the metrics over all folds

    :param total_divergence_metrics: Hash-map containing the sum of all divergence metrics of runs so far
    :param total_folds: Total number of folds (runs) where a clustering model was built and evaluated
    """
    average_kl_divergence = total_divergence_metrics["total_kl_divergence"] / total_folds
    average_js_divergence = total_divergence_metrics["total_js_divergence"] / total_folds
    average_tvd = total_divergence_metrics["total_tvd"] / total_folds

    print("Divergence Scores: quantify how the predicted high/low probability distribution differs from the actual distribution")
    print(f"Average Kullback-Leibler Divergence of the predicted scores from actual scores: {average_kl_divergence:.4f} bits")
    print(f"Average Jensen-Shannon Divergence between the predicted scores from actual scores: {average_js_divergence:.4f} bits")
    print(f"Average Total Variation Distance: {average_tvd:.4f}")
    print()

def kl_divergence(prob_distribution_a, prob_distribution_b):
    """
    Compute the Kullback-Leibler (KL) Divergence to quantifies difference between one probability distributions from
    another. Lower is better.

    :param prob_distribution_a: A probability distribution (predicted) - includes probability of a document being high-scoring and low-scoring
    :param prob_distribution_b: Another probability distribution (actual) - includes probability of a document being high-scoring and low-scoring
    :return: Kullback-Leibler Divergence between the two probability distributions

    A: predicted = [predicted high, predicted low]
    B:    actual = [actual high,    actual low]

    Divergence of the predicted from actual: A || B
    """
    possible_outcomes = len(prob_distribution_a)

    # Add small epsilon to avoid division by zero or ln(0)
    epsilon = 1e-8 # 10 ^ -8 = 0.00000001

    total_kl_divergence = 0
    for i in range(possible_outcomes):
        total_kl_divergence += prob_distribution_a[i] * log2(
            prob_distribution_a[i] / (prob_distribution_b[i] + epsilon) + epsilon
        )
    # first e to prevent division by 0, second one prevents case where numerator is 0 and we get ln(0) which is undefined

    return total_kl_divergence

# calculate the js divergence - symmetric
def js_divergence(prob_distribution_a, prob_distribution_b):
    """
    Compute the Jensen-Shannon (JS) Divergence to quantifies differences between probability distributions. Lower is better.
    :param prob_distribution_a: A probability distribution (predicted) - includes probability of a document being high-scoring and low-scoring
    :param prob_distribution_b: Another probability distribution (actual) - includes probability of a document being high-scoring and low-scoring
    :return: Jensen-Shannon Divergence between the two probability distributions
    """
    prob_distribution_a = np.array(prob_distribution_a)
    prob_distribution_b = np.array(prob_distribution_b)
    m = 0.5 * (prob_distribution_a + prob_distribution_b)
    return 0.5 * kl_divergence(prob_distribution_a, m) + 0.5 * kl_divergence(prob_distribution_b, m)

def total_variation_distance(prob_distribution_a, prob_distribution_b):
    """
    Compute total variation distance between probability distributions
    Lower is better: 0 means identical distributions and 1 means maximum divergence
    :param prob_distribution_a: A probability distribution (predicted) - includes probability of a document being high-scoring and low-scoring
    :param prob_distribution_b: Another probability distribution (actual) - includes probability of a document being high-scoring and low-scoring
    :return: Total Variation Distance between the two probability distributions
    """
    prob_distribution_a = np.array(prob_distribution_a)
    prob_distribution_b = np.array(prob_distribution_b)
    return 0.5 * np.sum(np.abs(prob_distribution_a - prob_distribution_b))
