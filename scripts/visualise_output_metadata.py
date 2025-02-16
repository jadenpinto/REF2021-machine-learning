import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import DATASETS_DIR, PROCESSED_REF2021_DIR, CS_OUTPUTS_METADATA, output_type
from utils.dataframe import log_dataframe


def get_outputs_metadata():
    processed_cs_outputs_metadata_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_REF2021_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_metadata = pd.read_csv(processed_cs_outputs_metadata_path)
    log_dataframe(cs_outputs_metadata)  # 7296 rows
    return cs_outputs_metadata

def plot_output_types(cs_outputs_metadata):
    output_type_counts = cs_outputs_metadata['Output type'].value_counts()

    # Plot Output types: "Journal article", "Conference contribution", and "Other"
    output_type_grouped_counts = {}

    for output_type_key, output_type_count in output_type_counts.items():
        type_of_output = output_type[output_type_key]
        print(f"Output Key={output_type_key}, Output Type = {type_of_output}, Count = {output_type_count}")

        if (type_of_output == "Journal article") or (type_of_output == "Conference contribution"):
            output_type_grouped_counts[type_of_output] = output_type_count
        else:
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
    plt.show()

cs_outputs_metadata = get_outputs_metadata()
plot_output_types(cs_outputs_metadata)

# TODO Initial data investigation -> can plot things etc (in first file)
# Visualise CS outputs
# Need one how maybe bar chart for every uni where it's split into composition of output types
# (stacked bar chart)

