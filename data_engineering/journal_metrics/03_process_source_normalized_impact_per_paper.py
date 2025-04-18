import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, RAW_DIR, SOURCE_NORMALIZED_IMPACT_PER_PAPER, SNIP

def main():
    snip_df = load_snip_df()
    processed_snip_df = process_snip_df(snip_df)
    write_snip_df(processed_snip_df)

def load_snip_df():
    snip_dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, RAW_DIR,
                                          SOURCE_NORMALIZED_IMPACT_PER_PAPER)

    snip_df = pd.read_excel(snip_dataset_path, sheet_name=0) # Sources
    return snip_df

def process_snip_df(raw_snip_df):
    processed_snip_df = raw_snip_df[
        raw_snip_df["Year"] == 2021
    ]

    processed_snip_df = processed_snip_df.filter(['Print ISSN', 'Electronic ISSN', 'SNIP'])

    print_issn = processed_snip_df[['SNIP', 'Print ISSN']].rename(columns={'Print ISSN': 'ISSN'})
    electronic_issn = processed_snip_df[['SNIP', 'Electronic ISSN']].rename(columns={'Electronic ISSN': 'ISSN'})

    processed_snip_df = pd.concat([print_issn, electronic_issn], ignore_index=True)
    processed_snip_df = processed_snip_df.dropna(subset=['ISSN'])

    return processed_snip_df

def write_snip_df(snip_df):
    snip_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR, SNIP)

    snip_df.to_parquet(snip_df_path, engine='fastparquet')

if __name__ == "__main__":
    main()
