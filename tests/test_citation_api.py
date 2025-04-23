from unittest.mock import patch, MagicMock
import pandas as pd
import importlib.util
import sys
from pathlib import Path

# Path to project root
project_root = Path(__file__).resolve().parent.parent
# Path to script containing functions to test
module_path = project_root / "data_engineering" / "output_metrics" / "01_scopus_citation_overview_api.py"
# Rename module so it does not start with a number
module_name = "scopus_citation_overview_api"

# Load module dynamically
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

# Access functions
get_citation_metadata = module.get_citation_metadata
extract_citation_metadata = module.extract_citation_metadata
process_citation_metadata = module.process_citation_metadata

# Set up dummy API key
module.elsevier_api_key = "ABCDE1234"


def test_extract_citation_metadata():
    sample_citation_metadata_api_payload = {'abstract-citations-response': {'identifier-legend': {'identifier': [{'@_fa': 'true', 'scopus_id': '85021243138'}]}, 'citeInfoMatrix': {'citeInfoMatrixXML': {'citationMatrix': {'cc': [{'$': '0'}, {'$': '0'}, {'$': '0'}, {'$': '1'}, {'$': '5'}, {'$': '4'}, {'$': '2'}], 'rangeCount': '12'}}}, 'citeColumnTotalXML': {'citeCountHeader': None}}}

    expected_extracted_citation_metrics = {'scopus_id': '85021243138', 'citation_counts_2014': 0, 'citation_counts_2015': 0, 'citation_counts_2016': 0, 'citation_counts_2017': 1, 'citation_counts_2018': 5, 'citation_counts_2019': 4, 'citation_counts_2020': 2, 'total_citations': 12}

    actual_extracted_citation_metrics = extract_citation_metadata(sample_citation_metadata_api_payload)

    assert isinstance(actual_extracted_citation_metrics, dict)
    assert expected_extracted_citation_metrics == actual_extracted_citation_metrics










"""
10.1145/3034786.3056106
{'abstract-citations-response': {'identifier-legend': {'identifier': [{'@_fa': 'true', 'scopus_id': '85021243138'}]}, 'citeInfoMatrix': {'citeInfoMatrixXML': {'citationMatrix': {'cc': [{'$': '0'}, {'$': '0'}, {'$': '0'}, {'$': '1'}, {'$': '5'}, {'$': '4'}, {'$': '2'}], 'rangeCount': '12'}}}, 'citeColumnTotalXML': {'citeCountHeader': None}}}
{'scopus_id': '85021243138', 'citation_counts_2014': 0, 'citation_counts_2015': 0, 'citation_counts_2016': 0, 'citation_counts_2017': 1, 'citation_counts_2018': 5, 'citation_counts_2019': 4, 'citation_counts_2020': 2, 'total_citations': 12}
10.1145/2579822
{'abstract-citations-response': {'identifier-legend': {'identifier': [{'@_fa': 'true', 'scopus_id': '84900319915'}]}, 'citeInfoMatrix': {'citeInfoMatrixXML': {'citationMatrix': {'cc': [{'$': '0'}, {'$': '0'}, {'$': '1'}, {'$': '2'}, {'$': '1'}, {'$': '2'}, {'$': '1'}], 'rangeCount': '7'}}}, 'citeColumnTotalXML': {'citeCountHeader': None}}}
{'scopus_id': '84900319915', 'citation_counts_2014': 0, 'citation_counts_2015': 0, 'citation_counts_2016': 1, 'citation_counts_2017': 2, 'citation_counts_2018': 1, 'citation_counts_2019': 2, 'citation_counts_2020': 1, 'total_citations': 7}
10.1007/s10601-016-9249-7
{'abstract-citations-response': {'identifier-legend': {'identifier': [{'@_fa': 'true', 'scopus_id': '84981503288'}]}, 'citeInfoMatrix': {'citeInfoMatrixXML': {'citationMatrix': {'cc': [{'$': '0'}, {'$': '0'}, {'$': '0'}, {'$': '0'}, {'$': '3'}, {'$': '3'}, {'$': '4'}], 'rangeCount': '10'}}}, 'citeColumnTotalXML': {'citeCountHeader': None}}}
{'scopus_id': '84981503288', 'citation_counts_2014': 0, 'citation_counts_2015': 0, 'citation_counts_2016': 0, 'citation_counts_2017': 0, 'citation_counts_2018': 3, 'citation_counts_2019': 3, 'citation_counts_2020': 4, 'total_citations': 10}
"""
