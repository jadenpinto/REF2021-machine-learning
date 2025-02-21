import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA

def get_cs_outputs_metadata():
    cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA)

    cs_outputs_df = pd.read_csv(cs_outputs_path)
    return cs_outputs_df

def get_cs_journal_article_metadata():
    cs_outputs_df = get_cs_outputs_metadata()
    cs_journal_article_metadata = cs_outputs_df[cs_outputs_df['Output type'] == "D"]
    return cs_journal_article_metadata
