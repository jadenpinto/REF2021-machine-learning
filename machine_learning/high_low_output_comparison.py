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

    # 5. Analyse Forensic science outputs
    analyse_forensic_science_outputs(high_scoring_outputs_df, low_scoring_outputs_df)


    print()



def analyse_top_citation_percentiles(high_scoring_outputs_df, low_scoring_outputs_df):
    # Count occurrences of each unique percentile in top_citation_percentile for both dataframes:
    high_scoring_top_percentile_counts = high_scoring_outputs_df['top_citation_percentile'].value_counts().sort_index()
    low_scoring_top_percentile_counts = low_scoring_outputs_df['top_citation_percentile'].value_counts().sort_index()

    # Log counts and percentage distribution of top citation percentiles
    print("High Scoring Outputs - top_citation_percentile counts:")
    print(high_scoring_top_percentile_counts)
    print("\nLow Scoring Outputs - top_citation_percentile counts:")
    print(low_scoring_top_percentile_counts)

    print("\nHigh Scoring Outputs - top_citation_percentile distribution (%):")
    print((high_scoring_top_percentile_counts / high_scoring_top_percentile_counts.sum() * 100))
    print("\nLow Scoring Outputs - top_citation_percentile distribution (%):")
    print((low_scoring_top_percentile_counts / low_scoring_top_percentile_counts.sum() * 100))

def analyse_citation_counts(high_scoring_outputs_df, low_scoring_outputs_df):
    print("\nHigh Scoring Outputs - Citation counts:")
    print(high_scoring_outputs_df['total_citations'].describe())
    print("\nLow Scoring Outputs - Citation counts:")
    print(low_scoring_outputs_df['total_citations'].describe())


def analyse_output_types(
    train, high_scoring_outputs_df, low_scoring_outputs_df, high_scoring_cluster_number, low_scoring_cluster_number
):
    high_scoring_output_type_counts = high_scoring_outputs_df['Output type'].value_counts().sort_index()
    low_scoring_output_type_counts = low_scoring_outputs_df['Output type'].value_counts().sort_index()

    # Log counts and percentage distribution of output types
    print("\nHigh Scoring Outputs - Output type counts:")
    print(high_scoring_output_type_counts)
    print("\nLow Scoring Outputs - Output type counts:")
    print(low_scoring_output_type_counts)

    print("\nHigh Scoring Outputs - Output type distribution (%):")
    print((high_scoring_output_type_counts / high_scoring_output_type_counts.sum() * 100))
    print("\nLow Scoring Outputs - Output type distribution (%):")
    print((low_scoring_output_type_counts / low_scoring_output_type_counts.sum() * 100))

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
    print("\nHigh Scoring Outputs - Number of additional authors:")
    print(high_scoring_outputs_df['Number of additional authors'].describe())
    print("\nLow Scoring Outputs - Number of additional authors:")
    print(low_scoring_outputs_df['Number of additional authors'].describe())


def analyse_interdisciplinary_research(high_scoring_outputs_df, low_scoring_outputs_df):
    # The value of this field is Yes if the output is interdisciplinary, if not, the field is left blank

    # Replace Nulls with No
    high_scoring_outputs_df.loc[:, 'Interdisciplinary'] = high_scoring_outputs_df['Interdisciplinary'].fillna('No')
    low_scoring_outputs_df.loc[:, 'Interdisciplinary'] = low_scoring_outputs_df['Interdisciplinary'].fillna('No')

    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Interdisciplinary'
    )

def analyse_forensic_science_outputs(high_scoring_outputs_df, low_scoring_outputs_df):
    # The value of this field is Yes if the output embodies research in forensic science, if not, the field is left blank

    # Replace Nulls with No
    high_scoring_outputs_df.loc[:, 'Forensic science'] = high_scoring_outputs_df['Forensic science'].fillna('No')
    low_scoring_outputs_df.loc[:, 'Forensic science'] = low_scoring_outputs_df['Forensic science'].fillna('No')

    log_counts_and_distribution_for_high_low_scoring_outputs(
        high_scoring_outputs_df, low_scoring_outputs_df, 'Forensic science'
    )



def log_counts_and_distribution_for_high_low_scoring_outputs(
    high_scoring_outputs_df, low_scoring_outputs_df, feature
): # todo call this method for the first few features
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



# todo - possibly Publisher, Year
