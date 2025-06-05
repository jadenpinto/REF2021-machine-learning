import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, RAW_DIR, SOURCE_NORMALIZED_IMPACT_PER_PAPER, SNIP

def main():
    """
    ETL pipeline for the SNIP dataset (used to fill-in values that are missing after API call)
    """
    snip_df = load_snip_df()
    processed_snip_df = process_snip_df(snip_df)
    write_snip_df(processed_snip_df)

def load_snip_df():
    """
    Read in the CWTS Journal metrics file as a pandas dataframe
    :return: Pandas dataframe of CWTS journal metrics including SNIP
    """
    snip_dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, RAW_DIR,
                                          SOURCE_NORMALIZED_IMPACT_PER_PAPER)

    snip_df = pd.read_excel(snip_dataset_path, sheet_name=0) # Sources
    return snip_df

def process_snip_df(raw_snip_df):
    """
    Filter CWTS Journal metrics for 2021 values of SNIP. Normalise data to 1NF by splitting each record (which has
    both electronic and print ISSN) into separate records with one ISSN per row.
    :param raw_snip_df: Pandas dataframe of CWTS journal metrics including SNIP
    :return: Pandas dataframe of ISSN and SNIP values
    """
    processed_snip_df = raw_snip_df[
        raw_snip_df["Year"] == 2021
    ]

    # The Raw SNIP file has multiple columns, filter for the ISSNs and the SNIP
    processed_snip_df = processed_snip_df.filter(['Print ISSN', 'Electronic ISSN', 'SNIP'])

    # Split the SNIP dataframe into two dataframes:
    # 1. Contains the print ISSN with the SNIP value. Rename print ISSN column as ISSN.
    print_issn = processed_snip_df[['SNIP', 'Print ISSN']].rename(columns={'Print ISSN': 'ISSN'})
    # 2. Contains the electronic ISSN with the SNIP value. Rename electronic ISSN column as ISSN
    electronic_issn = processed_snip_df[['SNIP', 'Electronic ISSN']].rename(columns={'Electronic ISSN': 'ISSN'})

    # Combine the two dataframes - this normalises it since now every row only has a single ISSN-SNIP pair
    # As opposed to having 2 ISSNs (print and electronic) with the SNIP value
    processed_snip_df = pd.concat([print_issn, electronic_issn], ignore_index=True)
    processed_snip_df = processed_snip_df.dropna(subset=['ISSN'])

    return processed_snip_df

def write_snip_df(snip_df):
    """
    Write the resulting SNIP dataframe as a parquet file
    :param snip_df: Pandas dataframe of ISSN and SNIP values
    """
    snip_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR, SNIP)

    snip_df.to_parquet(snip_df_path, engine='fastparquet')

if __name__ == "__main__":
    main()
