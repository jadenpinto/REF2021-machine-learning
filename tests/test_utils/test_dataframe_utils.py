import pandas as pd
from utils.dataframe import log_dataframe_column_types, split_df_on_null_field, log_null_values, delete_rows_by_values

def test_log_dataframe_column_types(capsys):
    """
    Test the utility function that logs data types of the dataframe's columns
    """
    df = pd.DataFrame({
        'name': ['Name1', 'Name2'],
        'age': [22, 17],
        'is_adult': [True, False]
    })

    log_dataframe_column_types(df)

    # Capture standard output
    captured = capsys.readouterr()
    expected_column_types_output = df.dtypes.to_string()

    assert expected_column_types_output in captured.out

def test_split_df_on_null_field():
    """
    Test the utility function that splits a dataframe based on if a specified column's value is null or not null
    """
    df = pd.DataFrame({
        'name': ['Name1', 'Name2', 'Name3', 'Name4'],
        'age': [16, None, 44, None]
    })

    null_df, not_null_df = split_df_on_null_field(df, 'age')

    # Dataframe of entries where age was null
    assert null_df['name'].tolist() == ['Name2', 'Name4']

    # Dataframe of entries where age was not null
    assert not_null_df['name'].tolist() == ['Name1', 'Name3']

def test_log_null_values(capsys):
    """
    Test the utility function that logs the number of nulls in the columns of a dataframe
    """
    df = pd.DataFrame({
        'name': ['Name1', 'Name2', 'Name3', 'Name4'],
        'age': [16, None, None, None]
    })

    log_null_values(df)

    captured = capsys.readouterr()
    assert "name    0" in captured.out
    assert "age     3" in captured.out

def test_delete_rows_by_values():
    """
    Test the utility function that deletes the rows where a specified column has a specified value
    """
    df = pd.DataFrame({
        'name': ['Name1', 'Name2', 'Name3', 'Name4', 'Name5'],
        'age': [18, 19, 99, 99, 29]
    })

    # Delete rows where age equals 99 or 29
    actual_filtered_df = delete_rows_by_values(df, 'age', [99, 29])

    expected_filtered_df = pd.DataFrame({
        'name': ['Name1', 'Name2'],
        'age': [18, 19]
    }).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual_filtered_df.reset_index(drop=True),
        expected_filtered_df
    )
