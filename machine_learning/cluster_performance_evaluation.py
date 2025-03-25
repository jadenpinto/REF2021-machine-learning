import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def get_cluster_evaluation_metrics(model, training_feature_array, predicted_training_labels):
    evaluation_metrics = {}

    evaluation_metrics["silhouette_score"] = silhouette_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["davies_bouldin_score"] = davies_bouldin_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["calinski_harabasz_score"] = calinski_harabasz_score(training_feature_array, predicted_training_labels)
    evaluation_metrics["inertia"] = model.inertia_ # Within-Cluster Sum of Squares (Inertia)

    return evaluation_metrics

def update_total_evaluation_metrics(total_evaluation_metrics, cluster_evaluation_metrics):
    total_evaluation_metrics["total_silhouette_score"] += cluster_evaluation_metrics["silhouette_score"]
    total_evaluation_metrics["total_davies_bouldin_score"] += cluster_evaluation_metrics["davies_bouldin_score"]
    total_evaluation_metrics["total_calinski_harabasz_score"] += cluster_evaluation_metrics["calinski_harabasz_score"]
    total_evaluation_metrics["total_inertia"] += cluster_evaluation_metrics["inertia"]

    return total_evaluation_metrics

def compute_cluster_evaluation_metrics(total_evaluation_metrics, total_folds):
    average_silhouette_score = total_evaluation_metrics["total_silhouette_score"] / total_folds
    average_davies_bouldin_score = total_evaluation_metrics["total_davies_bouldin_score"] / total_folds
    average_calinski_harabasz_score = total_evaluation_metrics["total_calinski_harabasz_score"] / total_folds
    average_inertia = total_evaluation_metrics["total_inertia"] / total_folds

    print(f"Average Silhouette Score = {average_silhouette_score}")
    print(f"Average Davies Bouldin Score = {average_davies_bouldin_score}")
    print(f"Average Calinski Harabasz Score = {average_calinski_harabasz_score}")
    print(f"Average Within-Cluster Sum of Squares (Inertia) = {average_inertia}")
