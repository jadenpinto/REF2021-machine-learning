# Research Excellence Framework 2021 - Data Modelling

## Description
Software developed as part of the major project to apply a data-driven approach using machine learning to predict the research output quality scores of individual submissions to the Research Excellence Framework (REF) 2021 Computer Science and Informatics Unit of Assessment (UOA).

## Prerequisites

1) `Python`: This project was developed using `Python 3.12.6`


2) Install all required dependencies (Python packages) listed in the requirements file using:

```
pip install -r requirements.txt
```

3) Obtain an Elsevier API Key from the [Elsevier Developer Portal](https://dev.elsevier.com/). For some Elsevier APIs used in the project, additional approval from the Elsevier team is needed, which can take up to a week. If you intend to run this code, please contact [Jaden Pinto](mailto:jcp11@aber.ac.uk) who will share his API key.


4) Create a `.env` file in the project root directory to store your Elsevier API key:

```
elsevier_api_key=your_elsevier_api_key_here
```

**Note**: The Elsevier API key is only required for refreshing or generating new data. All necessary outputs and journal metrics have already been pre-processed and stored as Parquet files in the datasets/ directory


## Test
Run the tests using the following command:
```
pytest tests/
```

## Scripts

### [Data Engineering](data_engineering)

#### [Journal Metrics](data_engineering/journal_metrics)

01_cs_journal_issn.py: ETL pipeline to obtain a file containing the ISSNs of journals of the outputs submitted to the CS UoA

02_scopus_serial_title_API.py: ETL pipeline to make API calls to the Scopus Serial Title API to retrieve the Scopus ID, SNIP, SJR, and Cite Score of journals (of the outputs submitted to the CS UoA)

03_process_source_normalized_impact_per_paper.py: ETL pipeline to process the CWTS Journal metrics file to obtain a DataFrame of normalised SNIP values that can be used to fill-in SNIPs of journals that were missing after unsuccessful API calls

04_process_scimago_journal_rank.py: ETL pipeline to process the SCImago Journal Rank file to obtain a DataFrame of normalised SJR values that can be used to fill-in SJRs of journals that were missing after unsuccessful API calls

05_handle_missing_journal_metrics.py: ETL pipeline to fill-in the journal metrics that were missing after unsuccessful Serial title API calls using the SNIP and SJR dataframes obtained by processing SNIP and SJR files in the previous two scripts.

06_scival_publication_API.py: Archived script. Originally (and incorrectly) used to obtain field-normalised metrics for journals rather than individual outputs, leading to mostly unsuccessful API calls.

#### [Output Metrics](data_engineering/output_metrics)

01_scopus_citation_overview_api.py: ETL pipeline to retrieve the citation counts of outputs submitted to the CS UoA using the Scopus Abstract Citations Count API

02_handle_missing_citations.py: ETL Pipeline to fill-in the citations of outputs submitted to the CS UoA that were missing after unsuccessful API calls

03_scival_publication_API.py: ETL pipeline to obtain and persist the field-normalised performance metrics of outputs submitted to the CS UoA: Top citation Percentile, field-weighted citation impact, field-weighted views impact, using the SciVal publication lookup API

### [Machine Learning](machine_learning)

create_cs_outputs_enriched_metadata.py: ETL pipeline to create dataframe of CS outputs with complete metadata including journal metrics, citation counts, and field-normalised output performance metrics.

check_for_normality.py: Contains graphical and statistical tests to assess whether features follow a normal distribution.

cs_output_results.py: Enhances the REF CS Output Quality Results by adding columns that indicate the number of high- and low-scoring outputs per university.

feature_engineering.py: Transforms features prior to use in the clustering model, including skewness correction, temporal standardisation, and inferring missing values.

size_constrained_clustering.py: Imported script implementing the deterministic annealing size-constrained clustering algorithm, with modifications applied.

train_test_clustering_models.py: Train and evaluate the clustering models using University-Based Leave-One-Out Cross-Validation

cluster_performance_evaluation.py: Compute and log the clustering model performance metrics - internal indices to assess cluster quality, and regression and statistical divergence metrics to asses cluster accuracy.

high_low_output_comparison.py: Feature analysis to identify the characteristics that distinguish high-quality research outputs from low-quality ones.
