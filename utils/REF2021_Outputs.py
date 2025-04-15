import os
import pandas as pd

from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA

def get_cs_outputs_metadata():
    """
    Return a dataframe of all CS outputs submitted to REF 2021
    :return: Pandas dataframe representing metadata of CS outputs submitted to REF 2021
    """
    cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_DIR, CS_OUTPUTS_METADATA)

    cs_outputs_df = pd.read_csv(cs_outputs_path)
    return cs_outputs_df

def get_cs_journal_article_metadata():
    """
    Return a dataframe of CS outputs filtered for journal articles only
    :return: Pandas dataframe representing metadata of CS journal articles submitted to REF 2021
    """
    cs_outputs_df = get_cs_outputs_metadata()
    cs_journal_article_metadata = cs_outputs_df[cs_outputs_df['Output type'] == "D"] # Filter for journal articles (D)
    return cs_journal_article_metadata
