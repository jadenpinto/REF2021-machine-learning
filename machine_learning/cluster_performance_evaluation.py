import numpy as np
from math import log2, sqrt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, r2_score

# Internal indices - No external information, evaluate

def get_cluster_evaluation_metrics(model, training_feature_array, predicted_training_labels):
    evaluation_metrics = {}

    evaluation_metrics["silhouette_score"] = silhouette_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["davies_bouldin_score"] = davies_bouldin_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["calinski_harabasz_score"] = calinski_harabasz_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["inertia"] = model.inertia_ # Within-Cluster Sum of Squares (Inertia)
    evaluation_metrics["bcss"] = model.compute_bcss(training_feature_array)  # Between-Cluster Sum of Squares

    return evaluation_metrics

def update_total_evaluation_metrics(total_evaluation_metrics, cluster_evaluation_metrics):
    total_evaluation_metrics["total_silhouette_score"] += cluster_evaluation_metrics["silhouette_score"]
    total_evaluation_metrics["total_davies_bouldin_score"] += cluster_evaluation_metrics["davies_bouldin_score"]
    total_evaluation_metrics["total_calinski_harabasz_score"] += cluster_evaluation_metrics["calinski_harabasz_score"]
    total_evaluation_metrics["total_inertia"] += cluster_evaluation_metrics["inertia"]
    total_evaluation_metrics["total_bcss"] += cluster_evaluation_metrics["bcss"]

    return total_evaluation_metrics

def compute_cluster_evaluation_metrics(total_evaluation_metrics, total_folds):
    average_silhouette_score = total_evaluation_metrics["total_silhouette_score"] / total_folds
    average_davies_bouldin_score = total_evaluation_metrics["total_davies_bouldin_score"] / total_folds
    average_calinski_harabasz_score = total_evaluation_metrics["total_calinski_harabasz_score"] / total_folds
    average_inertia = total_evaluation_metrics["total_inertia"] / total_folds
    average_bcss = total_evaluation_metrics["total_bcss"] / total_folds

    print(f"Average Silhouette Score = {average_silhouette_score}")
    print(f"Average Davies Bouldin Score = {average_davies_bouldin_score}")
    print(f"Average Calinski Harabasz Score = {average_calinski_harabasz_score}")
    print(f"Average Within-Cluster Sum of Squares (Inertia) = {average_inertia}")
    print(f"Average Between-Cluster Sum of Squares = {average_bcss}")

# External indices - Uses truth labels (proportion of high/low scoring outputs)

def compute_clustering_accuracy(
        actual_high_scoring_output_percentages,
        predicted_high_scoring_output_percentages,
        actual_low_scoring_output_percentage,
        predicted_low_scoring_output_percentage
    ):
    # Using both high and low percentages is redundant since one can be derived from the other
    # Instead, only use one to derive accuracy of model - High

    actual_high = np.array(actual_high_scoring_output_percentages)
    predicted_high = np.array(predicted_high_scoring_output_percentages)

    # Mean Absolute Error
    mae = np.mean(np.abs(actual_high - predicted_high))

    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual_high - predicted_high) ** 2))

    # Mean Absolute Percentage Error
    mape_values = []

    predictions = len(predicted_high_scoring_output_percentages)
    for prediction in range(predictions):
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

    # Lower is better!
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Mean Absolute Percentage Error: {mape:.3f}")

    # [-inf, 1] => 1 means Perfect predictions.
    # 0 means no relation at all between predicted and actual values
    print(f"R^2 score: {r2:.3f}")

def get_divergence_metrics(predicted_distribution, actual_distribution):
    divergence_metrics = {}

    divergence_metrics["kl_divergence"] = kl_divergence(predicted_distribution, actual_distribution)
    divergence_metrics["js_divergence"] = js_divergence(predicted_distribution, actual_distribution)
    divergence_metrics["tvd"] = total_variation_distance(predicted_distribution, actual_distribution)

    return divergence_metrics

def update_total_divergence_metrics(total_divergence_metrics, divergence_metrics):
    total_divergence_metrics["total_kl_divergence"] += divergence_metrics["kl_divergence"]
    total_divergence_metrics["total_js_divergence"] += divergence_metrics["js_divergence"]
    total_divergence_metrics["total_tvd"] += divergence_metrics["tvd"]

    return total_divergence_metrics

def compute_divergence_metrics(total_divergence_metrics, total_folds):
    average_kl_divergence = total_divergence_metrics["total_kl_divergence"] / total_folds
    average_js_divergence = total_divergence_metrics["total_js_divergence"] / total_folds
    average_tvd = total_divergence_metrics["total_tvd"] / total_folds

    print(f"Average Kullback-Leibler Divergence of the predicted scores from actual scores = {average_kl_divergence} bits")
    print(f"Average Jensen-Shannon Divergence between the predicted scores from actual scores = {average_js_divergence} bits")
    print(f"Average Total Variation Distance = {average_tvd}")

def kl_divergence(prob_distribution_a, prob_distribution_b):
    """
    p: predicted = [predicted high, predicted low]
    q:    actual = [actual high,    actual low]

    Divergence of the predicted from actual: P || Q
    """
    possible_outcomes = len(prob_distribution_a)

    # Add small epsilon to avoid division by zero or ln(0)
    epsilon = 1e-10

    total_kl_divergence = 0
    for i in range(possible_outcomes):
        total_kl_divergence += prob_distribution_a[i] * log2(
            prob_distribution_a[i] / (prob_distribution_b[i] + epsilon) + epsilon
        )
    # first e to prevent division by 0, second one prevents case where numerator is 0 and we get ln(0) which is undefined

    return total_kl_divergence

# calculate the js divergence - symmetric
def js_divergence(prob_distribution_a, prob_distribution_b):
    prob_distribution_a = np.array(prob_distribution_a)
    prob_distribution_b = np.array(prob_distribution_b)
    m = 0.5 * (prob_distribution_a + prob_distribution_b)
    return 0.5 * kl_divergence(prob_distribution_a, m) + 0.5 * kl_divergence(prob_distribution_b, m)

def total_variation_distance(prob_distribution_a, prob_distribution_b):
    prob_distribution_a = np.array(prob_distribution_a)
    prob_distribution_b = np.array(prob_distribution_b)
    return 0.5 * np.sum(np.abs(prob_distribution_a - prob_distribution_b))
