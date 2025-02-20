def log_dataframe(df):
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
    return df[~df[col].isin(values)]

def log_null_values(df):
    print(df.isna().sum())

# Column Types:
# print(sjr_df.dtypes)
