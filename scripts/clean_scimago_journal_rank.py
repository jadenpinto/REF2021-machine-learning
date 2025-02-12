import os
import pandas as pd

from utils.constants import DATASETS_DIR, IMPACT_FACTOR_EXPORTED_DIR, SCIMAGO_JOURNAL_RANK
from utils.dataframe import log_dataframe


def clean_sjr_dataset():
    # Path to SCImago journal rank file
    sjr_dataset_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, IMPACT_FACTOR_EXPORTED_DIR, SCIMAGO_JOURNAL_RANK)

    try:
        x = 1
        # Load the Excel file, skipping the first 4 lines -> 4th line will be used as the header
        # TODO sjr_df = pd.read_excel(sjr_dataset_path)

        # load excel, drop all but c1

        # log_dataframe(sjr_df)

        #return sjr_df

    except FileNotFoundError:
        print("File not found. Please make sure the file name and path are correct.")
    except Exception as e:
        print("An error occurred while reading the file:", str(e))

clean_sjr_dataset()


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