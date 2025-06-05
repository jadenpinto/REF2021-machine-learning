import os

from utils.REF2021_Outputs import get_cs_journal_article_metadata
from utils.constants import DATASETS_DIR, PROCESSED_DIR, CS_JOURNALS_ISSN

def main():
    """
    Pipeline to process journal ISSNs
    """
    process_cs_journal_ISSN()

def get_cs_journals_issn_df():
    """
    Filter the CS REF submissions for journal articles, and obtained the ISSNs as a pandas dataframe
    :return: Dataframe of CS Journal ISSNs
    """
    cs_journal_article_metadata = get_cs_journal_article_metadata()
    cs_journal_ISSN_df = cs_journal_article_metadata[["ISSN"]].drop_duplicates().dropna()
    return cs_journal_ISSN_df

def write_cs_journals_issn_df(cs_journal_ISSN_df):
    """
    Write the dataframe containing CS journal ISSNs to processed directory
    :param cs_journal_ISSN_df: Dataframe of CS Journal ISSNs
    """
    cs_journals_issn_df_path = os.path.join(os.path.dirname(__file__), "..", "..", DATASETS_DIR, PROCESSED_DIR,
                                          CS_JOURNALS_ISSN)

    cs_journal_ISSN_df.to_csv(cs_journals_issn_df_path, index=False)

def process_cs_journal_ISSN():
    """
    Extract all the journal ISSNs of submissions made to the CS UoA
    """
    cs_journal_ISSN_df = get_cs_journals_issn_df()
    write_cs_journals_issn_df(cs_journal_ISSN_df)

if __name__ == "__main__":
    main()
