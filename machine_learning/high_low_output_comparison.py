from utils.constants import output_type


def analyse_clusters(train, cluster_label_mapping):
    """

    The trained clusters are examined when Wrexham University's outputs (UKPRN=10007833) are used as the test set (fold),
    as it has only 9 outputs, the least among all universities.

    Compared to all other model configurations, this set-up has the largest number of data-points (outputs) in the training
    set.

    :param train:
    :param cluster_label_mapping:
    :return:
    """
    print(train.head().to_string())
    print(cluster_label_mapping)

    # Using the inferred clustering label mapping, figure out which cluster (0, 1) represents high and low scoring outputs
    for cluster_num, cluster_label in cluster_label_mapping.items():
        # Obtain the cluster number of high scoring outputs
        if cluster_label == 'high_scoring_outputs':
            high_scoring_cluster_number = cluster_num
        # Obtain the cluster number of low scoing outputs
        elif cluster_label == 'low_scoring_outputs':
            low_scoring_cluster_number = cluster_num

    # Split the dataframe into two based on the cluster label
    high_scoring_outputs_df = train[train['cluster'] == high_scoring_cluster_number]
    low_scoring_outputs_df = train[train['cluster'] == low_scoring_cluster_number]

    # Analyse features for High vs Low Scoring Output Patterns

    # 1. Analyse Top Citation percentile:
    analyse_top_citation_percentiles(high_scoring_outputs_df, low_scoring_outputs_df)

    # 2. Analyse Citation counts:
    analyse_citation_counts(high_scoring_outputs_df, low_scoring_outputs_df)

    # 3. Analyse Output Types
    analyse_output_types(
        train, high_scoring_outputs_df, low_scoring_outputs_df, high_scoring_cluster_number, low_scoring_cluster_number
    )

    # 4. Analyse Number of additional authors
    analyse_author_counts(high_scoring_outputs_df, low_scoring_outputs_df)

    # 5. Analyse Interdisciplinary research
    analyse_interdisciplinary_research(high_scoring_outputs_df, low_scoring_outputs_df)

    # 6. Analyse Forensic science outputs
    analyse_forensic_science_outputs(high_scoring_outputs_df, low_scoring_outputs_df)

    # 7. Analyse Criminology outputs
    analyse_criminology_outputs(high_scoring_outputs_df, low_scoring_outputs_df)

    # 8. Analyse Open access status
    analyse_open_access_status_outputs(high_scoring_outputs_df, low_scoring_outputs_df)

    # 9. Analyse Cross-referral requested
    analyse_cross_referral_requested_outputs(high_scoring_outputs_df, low_scoring_outputs_df)

    # 10. Incl factual info about significance
    analyse_significance_factual_info_outputs(high_scoring_outputs_df, low_scoring_outputs_df)

    # 11. Analyse SNIP
    analyse_snip(high_scoring_outputs_df, low_scoring_outputs_df)

    # 12. Analyse SJR
    analyse_sjr(high_scoring_outputs_df, low_scoring_outputs_df)

    # 13. Analyse Cite Score
    analyse_cite_score(high_scoring_outputs_df, low_scoring_outputs_df)

    # 14. Analyse Field Weighted Citation Impact
    analyse_field_weighted_citation_impact(high_scoring_outputs_df, low_scoring_outputs_df)

    # 15. Analyse Field Weighted Views Impact
    analyse_field_weighted_views_impact(high_scoring_outputs_df, low_scoring_outputs_df)

    print()



def analyse_top_citation_percentiles(high_scoring_outputs_df, low_scoring_outputs_df):
    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'top_citation_percentile'
    )

def analyse_citation_counts(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'total_citations'
    )


def analyse_output_types(
    train, high_scoring_outputs_df, low_scoring_outputs_df, high_scoring_cluster_number, low_scoring_cluster_number
):
    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Output type'
    )

    # For each output type, log what percentage os such outputs were high- and low-scoring:
    output_types = train['Output type'].unique()

    print("For each output type, percentage in high vs low scoring clusters:")
    for curr_output_type in output_types:
        current_output_type_df = train[train['Output type'] == curr_output_type]
        current_output_type_count = len(current_output_type_df)

        curr_type_high_count = len(current_output_type_df[current_output_type_df['cluster'] == high_scoring_cluster_number])
        curr_type_low_count = len(current_output_type_df[current_output_type_df['cluster'] == low_scoring_cluster_number])

        curr_output_type_high_percent = (curr_type_high_count / current_output_type_count * 100)
        curr_output_type_low_percent = (curr_type_low_count / current_output_type_count * 100)

        print(f"\nOutput Type: {output_type[curr_output_type]}")
        print(f"Total count: {current_output_type_count}")
        print(f"High scoring: {curr_type_high_count} ({curr_output_type_high_percent}%)")
        print(f"Low scoring: {curr_type_low_count} ({curr_output_type_low_percent}%)")


def analyse_author_counts(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Number of additional authors'
    )


def analyse_interdisciplinary_research(high_scoring_outputs_df, low_scoring_outputs_df):
    # The value of this field is Yes if the output is interdisciplinary, if not, the field is left blank

    # Replace Nulls with No
    high_scoring_outputs_df = replace_feature_nulls_with_no(high_scoring_outputs_df, 'Interdisciplinary')
    low_scoring_outputs_df = replace_feature_nulls_with_no(low_scoring_outputs_df, 'Interdisciplinary')

    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Interdisciplinary'
    )

def analyse_forensic_science_outputs(high_scoring_outputs_df, low_scoring_outputs_df):
    # The value of this field is Yes if the output embodies research in forensic science, if not, the field is left blank

    # Replace Nulls with No
    high_scoring_outputs_df = replace_feature_nulls_with_no(high_scoring_outputs_df, 'Forensic science')
    low_scoring_outputs_df = replace_feature_nulls_with_no(low_scoring_outputs_df, 'Forensic science')

    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Forensic science'
    )

def analyse_criminology_outputs(high_scoring_outputs_df, low_scoring_outputs_df):
    # The value of this field is Yes if the output embodies research in criminology, if not, the field is left blank

    # Replace Nulls with No
    high_scoring_outputs_df = replace_feature_nulls_with_no(high_scoring_outputs_df, 'Criminology')
    low_scoring_outputs_df = replace_feature_nulls_with_no(low_scoring_outputs_df, 'Criminology')

    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Criminology'
    )

def analyse_open_access_status_outputs(high_scoring_outputs_df, low_scoring_outputs_df):
    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Open access status'
    )

def analyse_cross_referral_requested_outputs(high_scoring_outputs_df, low_scoring_outputs_df):
    # If cross referral is requested, the value is the UOA to which the output was cross-referred to for advice
    # Is cross referral was not request, the field is left blank

    # Replace Nulls with No
    high_scoring_outputs_df = replace_feature_nulls_with_no(high_scoring_outputs_df, 'Cross-referral requested')
    low_scoring_outputs_df = replace_feature_nulls_with_no(low_scoring_outputs_df, 'Cross-referral requested')

    # Cannot call the log_counts_and_distribution_for_high_low_scoring_outputs() because sort_index()
    # doesn't work with numbers (UOA) and strings("No"
    high_scoring_outputs_feature_value_counts = high_scoring_outputs_df['Cross-referral requested'].value_counts()
    low_scoring_outputs_feature_value_counts = low_scoring_outputs_df['Cross-referral requested'].value_counts()

    print(f"High Scoring Outputs - Cross-referral requested counts:")
    print(high_scoring_outputs_feature_value_counts)
    print(f"\nLow Scoring Outputs - Cross-referral requested counts:")
    print(low_scoring_outputs_feature_value_counts)

    print(f"\nHigh Scoring Outputs - Cross-referral requested distribution (%):")
    print((high_scoring_outputs_feature_value_counts / high_scoring_outputs_feature_value_counts.sum() * 100))
    print(f"\nLow Scoring Outputs - Cross-referral requested distribution (%):")
    print((low_scoring_outputs_feature_value_counts / low_scoring_outputs_feature_value_counts.sum() * 100))

def analyse_significance_factual_info_outputs(high_scoring_outputs_df, low_scoring_outputs_df):
    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Incl factual info about significance'
    )

def analyse_snip(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'SNIP'
    )

def analyse_sjr(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'SJR'
    )

def analyse_cite_score(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Cite_Score'
    )

def analyse_field_weighted_citation_impact(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'field_weighted_citation_impact'
    )

def analyse_field_weighted_views_impact(high_scoring_outputs_df, low_scoring_outputs_df):
    log_continuous_feature_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'field_weighted_views_impact'
    )

def log_counts_and_distribution_for_high_low_scoring_outputs(
    high_scoring_outputs_df, low_scoring_outputs_df, feature
):
    # Count occurrences of each of the feature's values in both dataframes:
    high_scoring_outputs_feature_value_counts = high_scoring_outputs_df[feature].value_counts().sort_index()
    low_scoring_outputs_feature_value_counts = low_scoring_outputs_df[feature].value_counts().sort_index()

    # Log counts and percentage distribution of the values of the feature in both high- and low-scoring outputs
    print(f"High Scoring Outputs - {feature} counts:")
    print(high_scoring_outputs_feature_value_counts)
    print(f"\nLow Scoring Outputs - {feature} counts:")
    print(low_scoring_outputs_feature_value_counts)

    print(f"\nHigh Scoring Outputs - {feature} distribution (%):")
    print((high_scoring_outputs_feature_value_counts / high_scoring_outputs_feature_value_counts.sum() * 100))
    print(f"\nLow Scoring Outputs - {feature} distribution (%):")
    print((low_scoring_outputs_feature_value_counts / low_scoring_outputs_feature_value_counts.sum() * 100))

def log_continuous_feature_for_high_low_scoring_outputs(
    high_scoring_outputs_df, low_scoring_outputs_df, feature
):
    print(f"\nHigh Scoring Outputs - {feature}:")
    print(high_scoring_outputs_df[feature].describe())
    print(f"\nLow Scoring Outputs - {feature}:")
    print(low_scoring_outputs_df[feature].describe())

def replace_feature_nulls_with_no(df, feature):
    # Create a copy of the dataframe to avoid the SettingWithCopyWarning which occurs when modifying a copy of the slice
    # from a DataFrame.
    df = df.copy()

    # Replace the missing values of the feature with No
    df[feature] = df[feature].fillna('No')
    return df
