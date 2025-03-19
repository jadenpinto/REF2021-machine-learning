import os
import pandas as pd

from utils.constants import DATASETS_DIR, RAW_DIR, CS_RESULTS, MACHINE_LEARNING_DIR, CS_OUTPUTS_COMPLETE_METADATA


def main():
    cs_output_results_df = get_cs_output_results()

    # The total number of universities who have submitted CS outputs to REF2021: 90
    log_university_count(cs_output_results_df)

    cs_outputs_enriched_metadata = get_cs_outputs_enriched_metadata()
    enhanced_results_df = enhance_score_distribution(cs_output_results_df, cs_outputs_enriched_metadata)

    log_high_low_scoring_universities(enhanced_results_df)

def get_ref_results():
    results_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, RAW_DIR, CS_RESULTS)

    try:
        # Load the Excel file, skipping the first 6 lines -> The 7th line will be used as the header
        df = pd.read_excel(results_dataset_path, skiprows=6)
        return df

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

def get_cs_output_results():
    cs_results_df = get_ref_results()

    cs_output_results_df = filter_ref_outputs_for_cs(cs_results_df)
    return cs_output_results_df

def filter_ref_outputs_for_cs(cs_results_df):
    is_outputs_profile = cs_results_df['Profile'] == 'Outputs'
    cs_output_results_df = cs_results_df[
        is_outputs_profile
    ]

    return cs_output_results_df

def log_university_count(cs_output_results_df): # Institution code (UKPRN)
    university_count = cs_output_results_df['Institution code (UKPRN)'].nunique()
    print(f"The total number of universities who have submitted CS outputs to REF2021: {university_count}")
    assert university_count == cs_output_results_df.shape[0]

def get_cs_outputs_enriched_metadata():
    cs_outputs_enriched_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, MACHINE_LEARNING_DIR, CS_OUTPUTS_COMPLETE_METADATA
    )

    try:
        cs_outputs_enriched_metadata = pd.read_parquet(cs_outputs_enriched_metadata_path, engine='fastparquet')
        return cs_outputs_enriched_metadata

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))


def enhance_score_distribution(cs_output_results_df, cs_outputs_enriched_metadata):
    # Count total outputs per university
    total_outputs = cs_outputs_enriched_metadata.groupby('Institution UKPRN code').size().reset_index(
        name='total_university_outputs'
    )

    # Merge with cs_output_results_df on institution code
    enhanced_results_df = cs_output_results_df.merge(
        total_outputs,
        left_on='Institution code (UKPRN)',
        right_on='Institution UKPRN code',
        how='left'
    )

    # Calculate outputs per rating category
    for rating in ['4*', '3*', '2*', '1*', 'Unclassified']:
        column_name = f"{rating.replace('*', 'star').lower()}_outputs"
        output_count = (enhanced_results_df[rating] / 100) * enhanced_results_df['total_university_outputs']
        enhanced_results_df[column_name] = round(output_count)

    # Create high_scoring_outputs and low_scoring_outputs columns
    enhanced_results_df['high_scoring_outputs'] = enhanced_results_df['4star_outputs'] + enhanced_results_df['3star_outputs']
    enhanced_results_df['low_scoring_outputs'] = enhanced_results_df['2star_outputs'] + enhanced_results_df['1star_outputs'] + enhanced_results_df['unclassified_outputs']

    # Drop redundant column (that was added when merged)
    enhanced_results_df.drop(columns=['Institution UKPRN code'], inplace=True)

    return enhanced_results_df

def get_high_scoring_universities(enhanced_results_df):
    high_scoring_universities_df = enhanced_results_df[
        (enhanced_results_df["high_scoring_outputs"] > 0) &
        (enhanced_results_df["low_scoring_outputs"] == 0)
        ].sort_values(by="high_scoring_outputs", ascending=True)

    return high_scoring_universities_df[['Institution code (UKPRN)']]


def log_high_low_scoring_universities(enhanced_results_df):
    low_scoring_universities_df = enhanced_results_df[
        (enhanced_results_df["low_scoring_outputs"] > 0) &
        (enhanced_results_df["high_scoring_outputs"] == 0)
        ].sort_values(by="low_scoring_outputs", ascending=True)
    print(f"Number of universities where all CS outputs were scored low = {low_scoring_universities_df.shape[0]}")

    high_scoring_universities_df = enhanced_results_df[
        (enhanced_results_df["high_scoring_outputs"] > 0) &
        (enhanced_results_df["low_scoring_outputs"] == 0)
        ].sort_values(by="high_scoring_outputs", ascending=True)
    print(f"Number of universities where all CS outputs were scored high = {high_scoring_universities_df.shape[0]}")


if __name__ == "__main__":
    main()





"""
Camb - 20
Oxd  - 50
Imp -  10

[are they in 1 or in 0]
[if 1, elif 0]
[dont use else because they may not even be there]

Training includes: Camb, Oxd. Testing on Imp.
20+50+10 = 80 data points - check if more are in 0 or in 1. If more are in 1, then 1 is high scoring, otherwise 0 is high-scoring.

Since they all belong to one uni, you have a list of unis (UKPR code)
    - For datapoints having these UKPRs, how many of them are in cluster 0 and how many are in cluster 1
    - The cluster with a higher count of such datapoints, is said to be the high scoring cluster. 
    
possibly grouping by unis first
and then searching a lot more efficient. 
"""
