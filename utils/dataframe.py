def log_data_frame(df):
    # Display the first few rows to verify the result
    print(df.head().to_string())

    # Display the shape of the DataFrame
    print("\nNumber of rows and columns:", df.shape)

    # Columns
    print(df.columns)

