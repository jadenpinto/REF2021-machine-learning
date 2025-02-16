import os
import pandas as pd

from utils.constants import DATASETS_DIR, REF2021_EXPORTED_DIR, CS_UOA_RESULTS, OUTPUTS_METADATA, PROCESSED_REF2021_DIR, \
    CS_OUTPUTS_METADATA, output_type
from utils.dataframe import log_dataframe


def get_outputs_metadata():
    processed_cs_outputs_metadata_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_REF2021_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_metadata = pd.read_csv(processed_cs_outputs_metadata_path)
    log_dataframe(cs_outputs_metadata)  # 7296 rows
    return cs_outputs_metadata

def log_output_types(cs_outputs_metadata):
    output_type_counts = cs_outputs_metadata['Output type'].value_counts()

    for output_type_key, output_type_count in output_type_counts.items():
        type_of_output = output_type[output_type_key]
        print(f"Output Key={output_type_key}, Output Type = {type_of_output}, Count = {output_type_count}")

cs_outputs_metadata = get_outputs_metadata()
log_output_types(cs_outputs_metadata)

# TODO Initial data investigation -> can plot things etc (in first file)
# Visualise CS outputs