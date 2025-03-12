import os
import pandas as pd

from utils.constants import DATASETS_DIR, RAW_DIR, OUTPUTS_METADATA, PROCESSED_DIR, CS_OUTPUTS_METADATA
from utils.dataframe import log_dataframe


def get_cs_outputs_metadata():
    cs_outputs_metadata_path = os.path.join(
        os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA
    )

    cs_outputs_metadata = pd.read_csv(cs_outputs_metadata_path)
    return cs_outputs_metadata

def filter_cs_metadata_fields(cs_outputs_metadata):
    cs_outputs_metadata_fields = [
        'Institution UKPRN code', 'Institution name', 'Output type', 'Title', 'ISSN', 'DOI', 'Year',
        'Number of additional authors'
    ]
    # Possibly 'Citation count' -> but try to use this to back-fill citation API
    # Can remove later: Institution name (code is enough), Type, Title, Year
    # Institution UKPRN code: Results
    # ISSN, DOI: Join
    # Author count: Normalisation [Look into current literature, I think they use log]
    return cs_outputs_metadata[cs_outputs_metadata_fields]







def create_cs_dataset():
    cs_outputs_metadata = get_cs_outputs_metadata()
    cs_outputs_metadata = filter_cs_metadata_fields(cs_outputs_metadata)
    cs_outputs_metadata = cs_outputs_metadata
    # Use DOI
    # Use issn
    print(cs_outputs_metadata.head().to_string())
    print(cs_outputs_metadata.shape)


    return True

create_cs_dataset()

"""
Dataset represent CS outputs
"""