# Directories
DATASETS_DIR = "datasets"
RAW_DIR = "raw"
PROCESSED_DIR = "processed"
REFINED_DIR = "refined"

# Raw / Processed Files:
CS_RESULTS =  "REF2021_CS_Results.xlsx"
OUTPUTS_METADATA = "REF2021_Outputs_Metadata.xlsx"
CS_OUTPUTS_METADATA = "REF2021_CS_Outputs_Metadata.csv"
SCIMAGO_JOURNAL_RANK = "SCImago_Journal_Rank.csv"
CS_JOURNALS_ISSN = "REF2021_CS_Journals_ISSN.csv"
SOURCE_NORMALIZED_IMPACT_PER_PAPER = "CWTS_Journal_Indicators_SNIP.xlsx"
SJR = "SCImago_Journal_Rank.parquet"
SNIP = "SNIP.parquet"

# Refined Files
CS_JOURNAL_METRICS = "CS_Journal_Metrics.parquet"
CS_CITATION_METRICS = "CS_Citation_Metrics.parquet"
CS_OUTPUT_METRICS = "CS_Output_Metrics.parquet"

# Output Metadata
output_type = {
    "A": "Authored book",
    "B": "Edited book",
    "C": "Chapter in book",
    "D": "Journal article",
    "E": "Conference contribution", # Published in conference proceedings
    "F": "Patent",
    "G": "Software",
    "J": "Composition",
    "L": "Artefact",
    "M": "Exhibition",
    "N": "Research report for external body",
    "Q": "Digital or visual media",
    "U": "Working paper"
}
