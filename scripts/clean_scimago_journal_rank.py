import os
import pandas as pd

from utils.constants import DATASETS_DIR, IMPACT_FACTOR_EXPORTED_DIR, SCIMAGO_JOURNAL_RANK, IMPACT_FACTOR_CLEANED_DIR, \
    SCIMAGO_JOURNAL_RANK_CLEANED, REF2021_CLEANED_DIR, CS_OUTPUTS_METADATA
from utils.dataframe import log_dataframe, delete_rows_by_values

def clean_sjr_dataset():
    # Path to SCImago journal rank file
    sjr_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, IMPACT_FACTOR_EXPORTED_DIR, SCIMAGO_JOURNAL_RANK)

    try:
        sjr_df = pd.read_csv(
            sjr_dataset_path,
            delimiter=';',
            decimal=","
        )

        sjr_columns = ['Rank', 'Title', 'Issn', 'SJR']
        sjr_df = sjr_df[sjr_columns]

        log_dataframe(sjr_df)

        return sjr_df

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))


def count_issn_lengths(sjr_df):
    issn_lengths = sjr_df['Issn'].astype(str).apply(len)
    unique_issn_lengths = sorted(issn_lengths.unique())

    print(f"Possible Lengths of ISSNs in SJR dataframe={unique_issn_lengths}") # 1, 8, 18

def log_df_issn_lengths(sjr_df):
    # Create a temporary column to store the length of each Issn
    sjr_df['Issn_length'] = sjr_df['Issn'].astype(str).apply(len)

    # Filter DataFrame to include only rows where Issn_length is 1
    single_char_issn = sjr_df[sjr_df['Issn_length'] == 1]
    print("Logging records in DF where ISSN length is 1:")
    log_dataframe(single_char_issn)
    """
    28 rows
    Issn = "-" (in all 28 records)
    These journal do not have an Issn => Drop records where Issn = "-"
    """

    # Filter DataFrame to include only rows where Issn_length is 18
    eighteen_char_issn = sjr_df[sjr_df['Issn_length'] == 18]
    print("Logging records in DF where ISSN length is 18:")
    log_dataframe(eighteen_char_issn)
    """
    17740 rows
    Example: Issn = "15424863, 00079235"
    ISSN length is 18, if the field contains 2 ISSN values => Normalise to 1NF
    """

def handle_sjr_issn(sjr_df):
    sjr_df = drop_null_issn(sjr_df)
    sjr_df = convert_to_1nf(sjr_df)
    sjr_df = add_issn_hyphen(sjr_df)
    return sjr_df

def drop_null_issn(sjr_df):
    sjr_df = delete_rows_by_values(sjr_df, "Issn", ["-"])
    return sjr_df

def convert_to_1nf(sjr_df):
    sjr_df = sjr_df.assign(
        Issn=sjr_df['Issn'].str.split(', ')
    ).explode('Issn')

    return sjr_df

def add_issn_hyphen(sjr_df):
    sjr_df['Issn'] = sjr_df['Issn'].apply(
        lambda issn_str: issn_str[:4] + '-' + issn_str[4:]
    )

    return sjr_df

def write_cleaned_impact_factor(sjr_df):
    cleaned_impact_factor_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, IMPACT_FACTOR_CLEANED_DIR,
                                           SCIMAGO_JOURNAL_RANK_CLEANED)

    sjr_df.to_csv(cleaned_impact_factor_path, index=False)


sjr_df = clean_sjr_dataset() # NaN Count: SJR = 210, Rest = 0
count_issn_lengths(sjr_df)
log_df_issn_lengths(sjr_df)
sjr_df = handle_sjr_issn(sjr_df)
count_issn_lengths(sjr_df)
# Write:
# write_cleaned_impact_factor(sjr_df)


# Join -> maybe later move to another file
cleaned_cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REF2021_CLEANED_DIR,
                                       CS_OUTPUTS_METADATA)
cs_outputs_df = pd.read_csv(cleaned_cs_outputs_path)
journal_article_metadata = cs_outputs_df[cs_outputs_df['Output type'] == "D"]

cleaned_impact_factor_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, IMPACT_FACTOR_CLEANED_DIR,
                                           SCIMAGO_JOURNAL_RANK_CLEANED)
sjr_impact_df = pd.read_csv(cleaned_impact_factor_path)

print("-----------------------------JOIN:-------------------------------------------------------------------")
log_dataframe(journal_article_metadata)
log_dataframe(sjr_impact_df)

# journal_article_metadata [ISSN] join sjr_impact_df ['Issn']
print("joining...")
joined_df = pd.merge(
    journal_article_metadata, sjr_impact_df, left_on='ISSN', right_on='Issn', how='left'
)
log_dataframe(joined_df)

# Failed joins: Issn = None
print(joined_df.isna().sum()) # Only 313
# But here SJR is 317, explore why -> Issn is valid, but issn_length is None.
# Perhaps the original DF had invalid SJRs
# [I think this is it, quite a few, check which ones, and handle them - maybe hardcode if online]


"""
Example Output:
https://results2021.ref.ac.uk/outputs/1b7d4ae7-486f-455a-bf27-d490635acef2?page=1
where,
title: A bijective variant of the Burrows–Wheeler Transform using V-order
journal: Theoretical Computer Science
issn: 0304-3975

In SJR data, the row looks like this: (In column 1)
9480;20571;"Theoretical Computer Science";journal;"03043975";0

In SJR data, the headers:
Rank;Sourceid;Title;Type;Issn;SJR;SJR Best Quartile;H index;Total Docs. (2023);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas

Matching:
9480;20571;"Theoretical Computer Science";journal;"03043975";0
Rank;Sourceid;Title;Type;Issn;

remaining fields:
SJR;SJR Best Quartile;H index;Total Docs. (2023);Total Docs. (3years);Total Refs.;Total Cites (3years);Citable Docs. (3years);Cites / Doc. (2years);Ref. / Doc.;%Female;Overton;SDG;Country;Region;Publisher;Coverage;Categories;Areas
"""

# TODO
# Move Join to another file
# Initial data investigation -> can plot things etc (in first file)
