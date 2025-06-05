def log_dataframe(df):
    """
    Custom log function for a Pandas DataFrame, prints first 5 rows, shape, and columns
    :param df: Pandas DataFrame to log
    """
    # Display the first few rows to verify the result
    print(f"First 5 rows of the DF:")
    print(f"{df.head().to_string()}\n")

    # Display the shape of the DataFrame
    row_count, col_count = df.shape
    print(f"Number of rows={row_count}")
    print(f"Number of columns={col_count}\n")

    # Column Names
    print(f"Column Names:")
    print(f"{df.columns.values.tolist()}\n")

def delete_rows_by_values(df, col, values):
    """
    Remove all rows from a DataFrame where a specified column has a specific value.
    :param df: Pandas DataFrame
    :param col: Specified column name
    :param values: Specified value
    :return: Updated DataFrame with all rows where a specified column has a specific value are dropped
    """
    return df[~df[col].isin(values)]

def log_null_values(df):
    """
    Log the number of missing values in each column of a DataFrame
    :param df: Pandas DataFrame
    """
    print(df.isna().sum())

def log_dataframe_column_types(df):
    """
    Log the column data types of a DataFrame
    :param df: Pandas DataFrame
    """
    print(df.dtypes)

def split_df_on_null_field(df, field):
    """
    Split a Pandas DataFrame into two: one containing rows where a specified column is null, and another where it is not.
    :param df: Pandas DataFrame
    :param field: Specified column Name
    :return:
        null_field_df: DataFrame containing the rows from the original DataFrame where a specified column is null
        not_null_field_df: DataFrame containing the rows from the original DataFrame where a specified column is not null
    """
    field_is_null = df[field].isna()

    null_field_df = df[
        field_is_null
    ]
    not_null_field_df = df[
        ~field_is_null
    ]

    return null_field_df, not_null_field_df
