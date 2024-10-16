import pandas as pd

# Load the Excel and CSV files into DataFrames
daily_election_df = pd.read_excel('data/raw/Daily_Election_polymarket-price-data-04-01-2024-14-10-2024.xlsx')
tweets_csv_df = pd.read_csv('data/raw/tweets_Presidential_Election_data_Oct15_2024.csv')
tweets_xlsx_df = pd.read_excel('data/raw/tweets_Presidential_Election_data_Oct15_2024.xlsx')
tweets_scrapped_df = pd.read_excel('data/raw/Tweets_Scrapped_US_Elections_2024.xlsx')

# Define a helper function to display basic info for each DataFrame
def display_basic_info(df, df_name):
    print(f"--- {df_name} ---")
    print("Head of the DataFrame:")
    print(df.head(), "\n")
    print("Tail of the DataFrame:")
    print(df.tail(), "\n")
    print("Shape of the DataFrame:", df.shape)
    print(">> columns of he dataframe: ", df.columns)
    print("Number of missing values per column:")
    print(df.isna().sum(), "\n")

    # Check if there's a date column and display min and max dates if present
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_columns:
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                print(f"Min date in {date_col}: {df[date_col].min()}")
                print(f"Max date in {date_col}: {df[date_col].max()}\n")
            except Exception as e:
                print(f"Error processing date column {date_col}: {e}")
    else:
        print("No date columns detected.\n")

if __name__ == "__main__":
    # Display basic information for each DataFrame
    display_basic_info(daily_election_df, "Daily Election Polymarket Data")
    display_basic_info(tweets_csv_df, "Tweets Presidential Election Data (CSV)")
    display_basic_info(tweets_xlsx_df, "Tweets Presidential Election Data (XLSX)")
    display_basic_info(tweets_scrapped_df, "Tweets Scrapped US Elections 2024")
