import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA, output_type, FIGURES_DIR
from utils.dataframe import log_dataframe


def main():
    """
    Get CS outputs submission metadata, and visualise it to explore the different submission Types
    Log the total number of journals of the submissions
    Plot the distribution of output types for every university

    :return: None
    """
    cs_outputs_metadata = get_outputs_metadata()
    plot_output_types(cs_outputs_metadata)
    count_journals(cs_outputs_metadata)
    plot_university_output_distribution(cs_outputs_metadata) # stacked bar chart

def get_outputs_metadata():
    """
    Read REF2021_CS_Outputs_Metadata CSV File representing metadata of submissions made to CS UoA

    :return: cs_outputs_metadata - a pandas dataframe containing every submission made to the CS UOA with their metadata
    """
    processed_cs_outputs_metadata_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_metadata = pd.read_csv(processed_cs_outputs_metadata_path)
    log_dataframe(cs_outputs_metadata)  # 7296 rows
    return cs_outputs_metadata

def plot_output_types(cs_outputs_metadata):
    """
    Plot a pie chart of the different output types of the REF2021 submisssions

    :param cs_outputs_metadata: Pandas DF representing submissions made to the CS UOA
    :return: None
    """
    output_type_counts = cs_outputs_metadata['Output type'].value_counts()

    # Plot Output types
    # Group all different types into one of 3 categories: 1) Journal article 2) Conference contribution and 3) Other
    output_type_grouped_counts = {}

    for output_type_key, output_type_count in output_type_counts.items():
        type_of_output = output_type[output_type_key]
        print(f"Output Key={output_type_key}, Output Type = {type_of_output}, Count = {output_type_count}")

        if (type_of_output == "Journal article") or (type_of_output == "Conference contribution"):
            # Journal articles, and Conference contribution, are their own types of outputs
            output_type_grouped_counts[type_of_output] = output_type_count
        else:
            # All other outputs ground as the 'Other' Category
            output_type_grouped_counts['Other'] = output_type_grouped_counts.get('Other', 0) + output_type_count

    # Plot pie chart using grouped counts
    plt.pie(
        output_type_grouped_counts.values(),
        labels=output_type_grouped_counts.keys(),
        autopct='%1.1f%%'
    )
    plt.title('Distribution of Output Types')
    plt.legend(
        output_type_grouped_counts.keys(),
        title="Output Types",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    # Save an PNG in figures/ dir
    ref_cs_submission_types_distribution_path = os.path.join(
        os.path.dirname(__file__), "..", FIGURES_DIR, "ref_cs_submissions_type_distribution.png"
    )
    plt.savefig(ref_cs_submission_types_distribution_path)

    plt.show()

def count_journals(cs_outputs_metadata):
    """
    Log the total number of journals found in the submissions made to CS OuA

    :param cs_outputs_metadata: Pandas DF representing submissions made to the CS UOA
    :return: None
    """
    journal_names = cs_outputs_metadata['Volume title'].unique()
    print(f"The total number of journals in CS outputs metadata = {journal_names.size}") # 2734

def group_output_type(output_type_key):
    """
    Function returning output type based on the character key

    :param output_type_key: Character indication output type
    :return: Journal article if output is a journal article, Conference contribution for conference submissions, and Other
             otherwise
    """
    type_of_output = output_type[output_type_key]
    if (type_of_output == "Journal article") or (type_of_output == "Conference contribution"):
        return type_of_output
    else:
        return "Other"

def plot_university_pivot_data(pivot_data, filename):
    """
    Plot a stacked bar chart to visualise the distribution of research output types across universities
    :param pivot_data: Pandas DF pivot table with rows are universities, columns are output types, and
                       values are output counts
    :param filename: If specified, saves the plot to the figures dir/
    :return: None
    """
    # Sort universities by total output count
    pivot_data['Total'] = pivot_data.sum(axis=1)
    pivot_data = pivot_data.sort_values('Total', ascending=True)
    pivot_data = pivot_data.drop('Total', axis=1) # Delete temp total column

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.4)))  # Adjust height based on uni count

    # Create stacked bars
    bottom = np.zeros(len(pivot_data))
    for category in ['Journal article', 'Conference contribution', 'Other']: # Plot data types in this order
        if category in pivot_data.columns:
            values = pivot_data[category]
            ax.barh(pivot_data.index, values, left=bottom, label=category)
            bottom += values

    ax.set_title('Output Distribution by University')
    ax.set_xlabel('Number of Outputs')
    ax.set_ylabel('University')

    # Add plot legend in top left.
    ax.legend(title="Output Types", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add total count labels at the end of each bar
    for i in range(len(pivot_data)):
        total = sum(pivot_data.iloc[i])
        ax.text(total, i, f' Total: {total}', va='center')

    plt.tight_layout() # Tight layout, to fit all info on screen

    if filename:
        # Export plot to figures/
        ref_cs_submission_types_distribution_by_uni_path = os.path.join(
            os.path.dirname(__file__), "..", FIGURES_DIR, filename
        )
        plt.savefig(ref_cs_submission_types_distribution_by_uni_path)

    plt.show()
    plt.close()

def get_university_pivot_data(cs_outputs_metadata):
    """
    Convert the tabular cs output metadata to pivot table to get output counts by university and type

    :param cs_outputs_metadata: Pandas DF representing CS output metadata
    :return: Pivot table of CS output metadata
    """
    # Create a new column with grouped output types
    cs_outputs_metadata['Output Category'] = cs_outputs_metadata['Output type'].apply(group_output_type)

    # Create pivot table for the stacked bar chart - rows are universities, columns are output types, values are output counts
    pivot_data = pd.pivot_table(
        cs_outputs_metadata,
        values='Output type',
        index='Institution name',
        columns='Output Category',
        aggfunc='count',
        fill_value=0
    )
    pivot_data['Total'] = pivot_data.sum(axis=1) # Total outputs per university
    return pivot_data

def plot_university_output_distribution(cs_outputs_metadata):
    """
    Plot stacked bar charts visualising the research output distribution for universities - one for top and other for bottom
    half of universities based on total outputs
    :param cs_outputs_metadata: Pandas DF containing CS output metadata
    :return: None
    """
    pivot_data = get_university_pivot_data(cs_outputs_metadata)

    # Split unis into two halves based on total output counts
    university_count = len(pivot_data) // 2

    top_half_universities_by_output_counts = pivot_data.nlargest(university_count, 'Total')
    bottom_half_universities_by_output_counts = pivot_data.nsmallest(university_count, 'Total')

    # Plot top and bottom half of unis - pass in filenames to store in figures dir
    plot_university_pivot_data(
        top_half_universities_by_output_counts, "ref_cs_submissions_type_distribution_top_half.png"
    )
    plot_university_pivot_data(
        bottom_half_universities_by_output_counts, "ref_cs_submissions_type_distribution_bottom_half.png"
    )


if __name__ == "__main__":
    main()


"""
Investigating REF output metadata: print(cs_outputs_metadata.isna().sum())
# Non-English field, all were null (and no 'Yes' values) => All cs submissions were in English.
# Similarly Citations applicable = Yes for all => Citations were applicable for all submissions
# 'Delayed by COVID19' = Yes, for only 1 output
# 'Propose double weighting', and 'Is reserve output' all none
"""