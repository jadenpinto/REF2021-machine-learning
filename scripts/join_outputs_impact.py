import os
import pandas as pd

from utils.constants import (DATASETS_DIR, PROCESSED_IMPACT_FACTOR_DIR, PROCESSED_SCIMAGO_JOURNAL_RANK, REF2021_CLEANED_DIR,
                             CS_OUTPUTS_METADATA)
from utils.dataframe import log_dataframe

def get_cs_outputs_metadata():
    cleaned_cs_outputs_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, REF2021_CLEANED_DIR,
                                           CS_OUTPUTS_METADATA)

    cs_outputs_df = pd.read_csv(cleaned_cs_outputs_path)
    return cs_outputs_df

def get_journal_article_metadata(cs_outputs_df):
    journal_article_metadata = cs_outputs_df[cs_outputs_df['Output type'] == "D"]
    return journal_article_metadata

def get_sjr_impact_factor_df():
    processed_sjr_csv_path = os.path.join(os.path.dirname(__file__), "..", DATASETS_DIR, PROCESSED_IMPACT_FACTOR_DIR,
                                           PROCESSED_SCIMAGO_JOURNAL_RANK)

    sjr_impact_df = pd.read_csv(processed_sjr_csv_path)
    return sjr_impact_df

def join_journal_article_metadata_with_sjr():
    cs_outputs_df = get_cs_outputs_metadata()
    journal_article_metadata = get_journal_article_metadata(cs_outputs_df)
    sjr_impact_df = get_sjr_impact_factor_df()

    log_dataframe(journal_article_metadata) # Shape: 5573x40
    log_dataframe(sjr_impact_df)            # Shape: 46877x4

    # Join: journal_article_metadata [ISSN], sjr_impact_df ['Issn']
    joined_df = pd.merge(
        journal_article_metadata, sjr_impact_df, left_on='ISSN', right_on='Issn', how='left'
    )
    log_dataframe(joined_df)               # Shape: 5573x44
    return joined_df

def log_failed_joins(sjr_joined_df):
    # Failed joins: Records where Issn = None
    failed_join_count = sjr_joined_df['Issn'].isna().sum() #  313 failed joins. [successful joins = 5573-313 = 5260]
    print(f"Count of output records whose ISSN could not be found in journal metadata = {failed_join_count}")

    # Invalid SJR: Issn is Valid (successful join) but SJR is None
    joins_with_no_sjr = sjr_joined_df[
        ~sjr_joined_df['Issn'].isnull() & sjr_joined_df["SJR"].isna()
    ]
    print(f"Number of output records with successful ISSN joins but invalid SJR = {joins_with_no_sjr.shape[0]}")
    print(joins_with_no_sjr.to_string()) # 4x44 (4 such records where Issn is Valid but SJR is None)

def join_outputs_metadata_with_sjr():
    cs_outputs_df = get_cs_outputs_metadata()
    sjr_impact_df = get_sjr_impact_factor_df()

    log_dataframe(cs_outputs_df)            # Shape: 7296x40
    log_dataframe(sjr_impact_df)            # Shape: 46877x4

    joined_df = pd.merge(
        cs_outputs_df, sjr_impact_df, left_on='ISSN', right_on='Issn', how='left'
    )
    log_dataframe(joined_df)               # Shape: 7296x44

    # Failed joins: Issn = None
    print(joined_df.isna().sum()) # Issn = 1665. 1665 failed joins. [successful joins = 7296-1665 = 5631]
    # So, besides journal articles, a few entries of other types have an SJR impact [5631 - 5260 = 371]
    # I could visualise this, to see, of all records that were successfully joined, plot pie chart of output type

    return joined_df

journal_metadata_with_sjr = join_journal_article_metadata_with_sjr()
outputs_with_sjr = join_outputs_metadata_with_sjr()

log_failed_joins(journal_metadata_with_sjr)

"""
Also now that I had split up records earlier
Check how many duplicate article titles [dont search for journal titles since you can have multiple journals]
Actually no need to dedup, since you're joining all articles with journals using issn, so each gets joined to only one journal
"""

"""
Joined but no SJR:

      Institution UKPRN code             Institution name Main panel  Multiple submission letter  Multiple submission name  Joint submission Output type                                                                                                                                    Title_x Place Publisher                                  Volume title Volume   Issue First page Article number ISBN       ISSN                     DOI Patent number     Month    Year                            URL  Number of additional authors  Non-English Interdisciplinary Forensic science Criminology  Propose double weighting  Is reserve output                            Research group Open access status Citations applicable  Citation count  Cross-referral requested Supplementary information Delayed by COVID19                                REF2ID  Incl sig material before 2014  Incl reseach process  Incl factual info about significance     Rank                                                      Title_y       Issn  SJR  Issn_length
2147                10007772  Edinburgh Napier University          B                         NaN                       NaN               NaN           D  Employing a Machine Learning Approach to Detect Combined Internet of Things Attacks Against Two Objective Functions Using a Novel Dataset   NaN       NaN           Security and Communication Networks   2020     NaN          1        2804291  NaN  1939-0114    10.1155/2020/2804291           NaN  February  2020.0                            NaN                           2.0          NaN               NaN              NaN         NaN                       NaN                NaN                                       NaN          Compliant                  Yes             2.0                       NaN                       NaN                NaN  be658bb8-17cb-4b57-8eed-005c1f303820                              0                     0                                     0  29123.0           Security and Communication Networks (discontinued)  1939-0114  NaN         18.0
2169                10007803     University of St Andrews          B                         NaN                       NaN               NaN           D                                                                                            End-to-end mobility for the internet using ILNP   NaN       NaN  Wireless Communications and Mobile Computing   2019     NaN          1        7464179  NaN  1530-8669    10.1155/2019/7464179           NaN     April  2019.0                            NaN                           1.0          NaN               NaN              NaN         NaN                       NaN                NaN                               B - Systems          Compliant                  Yes             1.0                       NaN                       NaN                NaN  c6174a65-03de-4d5c-811f-ba036617ad78                              0                     1                                     1  29162.0  Wireless Communications and Mobile Computing (discontinued)  1530-8669  NaN         18.0
3775                10007786        University of Bristol          B                         NaN                       NaN               NaN           D                                                                                        Onboard evolution of understandable swarm behaviors   NaN       NaN                  Advanced Intelligent Systems      1       6        NaN        1900031  NaN  2640-4567  10.1002/aisy.201900031           NaN      July  2019.0                            NaN                           3.0          NaN               NaN              NaN         NaN                       NaN                NaN  A - Artificial Intelligence and Autonomy          Compliant                  Yes             NaN                       NaN                       NaN                NaN  46207927-14b4-4c0a-bf4b-dc1f736793b6                              0                     0                                     1  28958.0                                 Advanced Intelligent Systems  2640-4567  NaN          8.0
5242                10007150       The University of Kent          B                         NaN                       NaN               NaN           D                                                       Trust Management for Public Key Infrastructures: Implementing the X.509 Trust Broker   NaN       NaN           Security and Communication Networks   2017  690714          1        6907146  NaN  1939-0114    10.1155/2017/6907146           NaN  February  2017.0  https://kar.kent.ac.uk/60311/                           6.0          NaN               NaN              NaN         NaN                       NaN                NaN                                       NaN          Compliant                  Yes             0.0                       NaN                       NaN                NaN  f7d0421c-8b0c-4f7b-bf68-b43dc00be344                              0                     0                                     1  29123.0           Security and Communication Networks (discontinued)  1939-0114  NaN         18.0

SJR:
Security and Communication Network -> 0.494 (as of 2022)
Wireless Communications and Mobile Computing -> 0.445 (as of 2022)
Advanced Intelligent Systems -> No Data
"""