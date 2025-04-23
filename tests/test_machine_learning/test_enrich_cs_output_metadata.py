import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from machine_learning.create_cs_outputs_enriched_metadata import filter_cs_metadata_fields, enrich_metadata_with_journal_metrics, enrich_metadata_with_output_metrics, enrich_cs_outputs_metadata

@pytest.fixture
def cs_outputs_metadata():
    """
    CS outputs metadata dataframe consisting of a single CS output (before it is enriched)
    """
    return pd.DataFrame({
        "Institution UKPRN code": [10007794],
        "Institution name": ["University of Glasgow"],
        "Output type": ["D"],
        "Title": ['"Almost-stable" matchings in the Hospitals / Residents problem with Couples'],
        "Volume title": ["Constraints"],
        "Place": [None],
        "Publisher": [None],
        "ISSN": ["1383-7133"],
        "DOI": ["10.1007/s10601-016-9249-7"]
    })

def test_filter_cs_metadata_fields():
    cs_outputs_metadata_json = {
        "Institution UKPRN code": [10007760, 10007855],
        "Institution name": ["Birkbeck College", "Swansea University / Prifysgol Abertawe"],
        "Main panel": ["B", "B"],
        "Output type": ["E", "D"],
        "Title": ["J-Logic: Logical Foundations for JSON Querying", "Parity Games and Propositional Proofs"],
        "Volume title": [
            "Proceedings of the 36th ACM SIGMOD-SIGACT-SIGAI Symposium on Principles of Database Systems - PODS '17",
            "ACM Transactions on Computational Logic"
        ],
        "Place": [None, None],
        "Publisher": [None, None],
        "ISSN": [None, "1529-3785"],
        "DOI": ["10.1145/3034786.3056106", "10.1145/2579822"],
        "Year": [2017.0, 2014.0],
        "Number of additional authors": [2.0, None],
        "Interdisciplinary": [None, None],
        "Forensic science": [None, None],
        "Criminology": [None, None],
        "Research group": ["3 - Knowledge Representation and Data Management", None],
        "Open access status": [None, "Out of scope for open access requirements"],
        "Cross-referral requested": [None, None],
        "Delayed by COVID19": [None, None],
        "Incl sig material before 2014": [0, 0],
        "Incl reseach process": [0, 0],
        "Incl factual info about significance": [1, 1],
        "Additional column for testing": ["value_one", "value_two"],
        "REF2ID": [1, 2]
    }
    cs_outputs_metadata_df = pd.DataFrame(cs_outputs_metadata_json)

    filted_cs_outputs_metadata = filter_cs_metadata_fields(cs_outputs_metadata_df)

    expected_columns = [
        'Institution UKPRN code', 'Institution name', 'Output type', 'Title', 'Volume title',
        'Place', 'Publisher', 'ISSN', 'DOI', 'Year',
        'Number of additional authors', 'Interdisciplinary', 'Forensic science', 'Criminology',
        'Research group', 'Open access status', 'Cross-referral requested', 'Delayed by COVID19',
        'Incl sig material before 2014', 'Incl reseach process', 'Incl factual info about significance'
    ]

    assert list(filted_cs_outputs_metadata.columns) == expected_columns

def test_enrich_metadata_with_journal_metrics(cs_outputs_metadata):
    # Journal metrics dataframe containing metadata of three different journals, with ISSN: "1383-7133", "2168-2305", "2213-235X"
    cs_journal_metrics_mock = pd.DataFrame({
        "ISSN": ["1383-7133", "2168-2305", "2213-235X"],
        "Scopus_ID": ["24175", 2417544, "21100853871"],
        "SNIP": [1.299, 1.595, None],
        "SJR": [0.626, 0.622, 0.584],
        "Cite_Score": [7.5, 7.2, 7.4]
    })

    with patch("machine_learning.create_cs_outputs_enriched_metadata.load_cs_journal_metrics_df", return_value=cs_journal_metrics_mock):
        # Enrich the output metrics for the CS output
        # The output is published in a journal with ISSN 1383-7133
        # To the metadata of the output, include the journal metrics of the journal having ISSN value 1383-7133
        cs_outputs_metadata_journal_metrics = enrich_metadata_with_journal_metrics(cs_outputs_metadata)

    # Assertions, check that the CS output was enriched with the journal metrics of the journal it was published it
    assert cs_outputs_metadata_journal_metrics.loc[0, "ISSN"] == "1383-7133"
    assert cs_outputs_metadata_journal_metrics.loc[0, "Scopus_ID"] == "24175"
    assert cs_outputs_metadata_journal_metrics.loc[0, "SNIP"] == 1.299
    assert cs_outputs_metadata_journal_metrics.loc[0, "SJR"] == 0.626
    assert cs_outputs_metadata_journal_metrics.loc[0, "Cite_Score"] == 7.5

def test_enrich_metadata_with_output_metrics():
    # The metadata of one CS output
    cs_outputs_metadata = pd.DataFrame(
        {
            'Institution UKPRN code': [10007794],
            'Institution name': ['University of Glasgow'],
            'Output type': ['D'],
            'Title': ['"Almost-stable" matchings in the Hospitals / Residents problem with Couples'],
            'Volume title': ['Constraints'],
            'Place': [None],
            'Publisher': [None],
            'ISSN': ['1383-7133'],
            'Scopus_ID': ['24175'],
            'SNIP': [1.299],
            'SJR': [0.626],
            'Cite_Score': [7.5],
            'DOI': ['10.1007/s10601-016-9249-7']
        }
    )

    # The citation metadata of the same CS output
    cs_citation_metadata_df = pd.DataFrame(
        {
            'scopus_id': ['84981503288'],
            'citation_counts_2014': [0.0],
            'citation_counts_2015': [0.0],
            'citation_counts_2016': [0.0],
            'citation_counts_2017': [0.0],
            'citation_counts_2018': [3.0],
            'citation_counts_2019': [3.0],
            'citation_counts_2020': [4.0],
            'total_citations': [10.0],
            'DOI': ['10.1007/s10601-016-9249-7']
        }
    )

    # The field normalised metadata for the same CS output
    cs_output_metrics_df = pd.DataFrame(
        {
            'field_weighted_citation_impact': [1.39],
            'top_citation_percentile': [25.0],
            'field_weighted_views_impact': [1.0826442],
            'scopus_id': ['84981503288']
        }
    )

    expected_cs_outputs_metadata_citation_output_metrics_df = pd.DataFrame(
        {
            'Institution UKPRN code': [10007794],
            'Institution name': ['University of Glasgow'],
            'Output type': ['D'],
            'Title': ['"Almost-stable" matchings in the Hospitals / Residents problem with Couples'],
            'Volume title': ['Constraints'],
            'Place': [None],
            'Publisher': [None],
            'ISSN': ['1383-7133'],
            'Scopus_ID': ['24175'],
            'SNIP': [1.299],
            'SJR': [0.626],
            'Cite_Score': [7.5],
            'DOI': ['10.1007/s10601-016-9249-7'],
            'scopus_id': ['84981503288'],
            'citation_counts_2014': [0.0],
            'citation_counts_2015': [0.0],
            'citation_counts_2016': [0.0],
            'citation_counts_2017': [0.0],
            'citation_counts_2018': [3.0],
            'citation_counts_2019': [3.0],
            'citation_counts_2020': [4.0],
            'total_citations': [10.0],
            'field_weighted_citation_impact': [1.39],
            'top_citation_percentile': [25.0],
            'field_weighted_views_impact': [1.0826442]
        }
    )

    with patch("machine_learning.create_cs_outputs_enriched_metadata.load_cs_citation_metadata_df", return_value=cs_citation_metadata_df), \
         patch("machine_learning.create_cs_outputs_enriched_metadata.load_cs_output_metrics_df", return_value=cs_output_metrics_df):

        actual_cs_outputs_metadata_citation_output_metrics_df = enrich_metadata_with_output_metrics(cs_outputs_metadata)

        pd.testing.assert_frame_equal(
            actual_cs_outputs_metadata_citation_output_metrics_df.reset_index(drop=True),
            expected_cs_outputs_metadata_citation_output_metrics_df.reset_index(drop=True)
        )


def test_enrich_cs_outputs_metadata_calls_enrichment_methods():
    """
    since the inner methods - enrich_metadata_with_journal_metrics, enrich_metadata_with_output_metrics are already tested
    This test checks whether these methods are called when enriching CS outputs
    """
    mock_input_df = MagicMock(name="mock_input_df")
    mock_mid_result = MagicMock(name="mock_mid_result")
    mock_final_result = MagicMock(name="mock_final_result")

    with patch("machine_learning.create_cs_outputs_enriched_metadata.enrich_metadata_with_journal_metrics", return_value=mock_mid_result) as mock_journal_metrics, \
         patch("machine_learning.create_cs_outputs_enriched_metadata.enrich_metadata_with_output_metrics", return_value=mock_final_result) as mock_output_metrics:

        result = enrich_cs_outputs_metadata(mock_input_df)

        # Assert that the CS output was enhanced with journal metadata
        mock_journal_metrics.assert_called_once_with(mock_input_df)
        # Assert that the CS output was enhanced with output metadata - citation + field normalised metrics
        mock_output_metrics.assert_called_once_with(mock_mid_result)
        assert result == mock_final_result
