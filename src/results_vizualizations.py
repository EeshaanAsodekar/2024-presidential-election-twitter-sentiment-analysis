import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_abs_correl_heatmap():
    '''
    Plots heatmap of the sentiment scores, polymarket odds, and the 
    market indexes.
    NOTE: this is plots absoloutes, not pct changes
    TODO: create a new fucntion called plot_pct_correl_heatmap which
    plots the pct changes in the scores and others' correl
    '''
    # Load dataframes
    ### final {9 x columns} dataframes
    df_odds = pd.read_excel(r"results\computed_results.xlsx")
    df_pos_score = pd.read_csv(r"results\positive_stats.csv")
    df_neg_score = pd.read_csv(r"results\negative_stats.csv")

    # Convert 'Date' columns to datetime in all dataframes
    df_odds['Date'] = pd.to_datetime(df_odds['Date'])
    df_pos_score['Date'] = pd.to_datetime(df_pos_score['Date'])
    df_neg_score['Date'] = pd.to_datetime(df_neg_score['Date'])

    print("------------------------------------------------")
    print(df_odds.head(7))
    print(df_odds.columns)
    print("------------------------------------------------")

    print("------------------------------------------------")
    print(df_pos_score.head(7))
    print(df_pos_score.columns)
    print("------------------------------------------------")

    print("------------------------------------------------")
    print(df_neg_score.head(7))
    print(df_neg_score.columns)
    print("------------------------------------------------")


    # Merge the dataframes on 'Date'
    df_merged = df_odds.merge(df_pos_score, on='Date').merge(df_neg_score, on='Date', suffixes=('_pos', '_neg'))

    # Select columns of interest
    neg_columns_of_interest = [
        'Date', 
        'Donald Trump', 
        'Kamala Harris',
        'negative_trump_score_mean', 
        'negative_trump_score_median',
        # 'negative_trump_score_<lambda_0>', 
        # 'negative_trump_score_<lambda_1>',
        'negative_harris_score_mean', 
        'negative_harris_score_median',
        # 'negative_harris_score_<lambda_0>', 
        # 'negative_harris_score_<lambda_1>',
        'calculated_neg_trump_score_mean', 
        'calculated_neg_trump_score_median',
        # 'calculated_neg_trump_score_<lambda_0>',
        # 'calculated_neg_trump_score_<lambda_1>',
        'calculated_neg_harris_score_mean',
        'calculated_neg_harris_score_median',
        # 'calculated_neg_harris_score_<lambda_0>',
        # 'calculated_neg_harris_score_<lambda_1>', 
        'abs_neg_trump_score_mean',
        'abs_neg_trump_score_median', 
        # 'abs_neg_trump_score_<lambda_0>',
        # 'abs_neg_trump_score_<lambda_1>', 
        'abs_neg_harris_score_mean',
        'abs_neg_harris_score_median', 
        # 'abs_neg_harris_score_<lambda_0>',
        # 'abs_neg_harris_score_<lambda_1>'
    ]

    df_mkt_data = pd.read_csv("data/raw/market_data.csv")
    print(df_mkt_data.head())

    df_analysis = df_merged[neg_columns_of_interest]
    print(df_analysis.head())

    # Ensure both Date columns are of the same data type (datetime64[ns])
    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
    df_mkt_data['Date'] = pd.to_datetime(df_mkt_data['Date'])

    # Perform the left merge
    df_analysis_mkt_data = df_analysis.merge(df_mkt_data, on='Date', how='left')

    # Display the merged DataFrame
    print("****************\nmerged + mkt dataset\n")
    print(df_analysis_mkt_data.head())
    print(df_analysis_mkt_data.shape)

    print("****************\nFinal merged dataset\n")
    print(df_analysis.head(7))
    print(df_analysis.shape)
    print(df_analysis.columns)


    ### CORREL HEATMAP
    # Compute the correlation matrix
    # corr = df_analysis.drop('Date', axis=1).corr()
    corr = df_analysis_mkt_data.drop('Date', axis=1).corr()

    # Visualize the correlation matrix
    plt.figure(figsize=(40, 30))
    sns.set(font_scale=0.6)
    # Create the heatmap with tick labels removed initially
    heatmap = sns.heatmap(
        corr, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        annot_kws={"size": 10},
        xticklabels=True, yticklabels=True
    )

    # Move the x-axis labels to the top
    heatmap.xaxis.set_ticks_position('top')  # Place x-axis labels on top
    plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability

    plt.title('Correlation Matrix between Odds and Sentiment Scores')
    plt.savefig("results/mean_medians_correl.png")
    plt.show()

def plot_abs_time_series():
    '''
    Plots time series of the sentiment scores, polymarket odds, and the 
    market indexes.
    NOTE: this is plots absoloutes, not pct changes
    TODO: create a new fucntion called plot_abs_time_series which
    plots the pct changes in the scores and others' correl
    '''
    # Load dataframes
    ### final {9 x columns} dataframes
    df_odds = pd.read_excel(r"results\computed_results.xlsx")
    df_pos_score = pd.read_csv(r"results\positive_stats.csv")
    df_neg_score = pd.read_csv(r"results\negative_stats.csv")

    # Convert 'Date' columns to datetime in all dataframes
    df_odds['Date'] = pd.to_datetime(df_odds['Date'])
    df_pos_score['Date'] = pd.to_datetime(df_pos_score['Date'])
    df_neg_score['Date'] = pd.to_datetime(df_neg_score['Date'])

    print("------------------------------------------------")
    print(df_odds.head(7))
    print(df_odds.columns)
    print("------------------------------------------------")

    print("------------------------------------------------")
    print(df_pos_score.head(7))
    print(df_pos_score.columns)
    print("------------------------------------------------")

    print("------------------------------------------------")
    print(df_neg_score.head(7))
    print(df_neg_score.columns)
    print("------------------------------------------------")


    # Merge the dataframes on 'Date'
    df_merged = df_odds.merge(df_pos_score, on='Date').merge(df_neg_score, on='Date', suffixes=('_pos', '_neg'))

    # Select columns of interest
    neg_columns_of_interest = [
        'Date', 
        'Donald Trump', 
        'Kamala Harris',
        'negative_trump_score_mean', 
        'negative_trump_score_median',
        # 'negative_trump_score_<lambda_0>', 
        # 'negative_trump_score_<lambda_1>',
        'negative_harris_score_mean', 
        'negative_harris_score_median',
        # 'negative_harris_score_<lambda_0>', 
        # 'negative_harris_score_<lambda_1>',
        'calculated_neg_trump_score_mean', 
        'calculated_neg_trump_score_median',
        # 'calculated_neg_trump_score_<lambda_0>',
        # 'calculated_neg_trump_score_<lambda_1>',
        'calculated_neg_harris_score_mean',
        'calculated_neg_harris_score_median',
        # 'calculated_neg_harris_score_<lambda_0>',
        # 'calculated_neg_harris_score_<lambda_1>', 
        'abs_neg_trump_score_mean',
        'abs_neg_trump_score_median', 
        # 'abs_neg_trump_score_<lambda_0>',
        # 'abs_neg_trump_score_<lambda_1>', 
        'abs_neg_harris_score_mean',
        'abs_neg_harris_score_median', 
        # 'abs_neg_harris_score_<lambda_0>',
        # 'abs_neg_harris_score_<lambda_1>'
    ]

    df_mkt_data = pd.read_csv("data/raw/market_data.csv")
    print(df_mkt_data.head())

    df_analysis = df_merged[neg_columns_of_interest]
    print(df_analysis.head())

    # Ensure both Date columns are of the same data type (datetime64[ns])
    df_analysis['Date'] = pd.to_datetime(df_analysis['Date'])
    df_mkt_data['Date'] = pd.to_datetime(df_mkt_data['Date'])

    # Perform the left merge
    df_analysis_mkt_data = df_analysis.merge(df_mkt_data, on='Date', how='left')

    # Display the merged DataFrame
    print("****************\nmerged + mkt dataset\n")
    print(df_analysis_mkt_data.head())
    print(df_analysis_mkt_data.columns)
    # print(df_analysis_mkt_data.shape)

    ## time series code
    score_columns = [
        'negative_trump_score_mean', 'negative_trump_score_median',
        'negative_harris_score_mean', 'negative_harris_score_median',
        'calculated_neg_trump_score_mean', 'calculated_neg_trump_score_median',
        'calculated_neg_harris_score_mean', 'calculated_neg_harris_score_median',
        'abs_neg_trump_score_mean', 'abs_neg_trump_score_median',
        'abs_neg_harris_score_mean', 'abs_neg_harris_score_median'
    ]

    # Define the market variables to plot
    market_columns = ['^RUA', '^GSPC', '^NDX', 'CL=F', 'NG=F']

    # Loop through each score column and plot against all market variables
    for score_col in score_columns:
        for market_col in market_columns:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Plot the sentiment score on the left y-axis
            ax1.plot(
                df_analysis_mkt_data['Date'], 
                df_analysis_mkt_data[score_col], 
                label=score_col, color='blue'
            )
            ax1.set_xlabel('Date')
            ax1.set_ylabel(score_col, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Create the second y-axis for the market variable
            ax2 = ax1.twinx()
            ax2.plot(
                df_analysis_mkt_data['Date'], 
                df_analysis_mkt_data[market_col], 
                label=market_col, color='red'
            )
            ax2.set_ylabel(market_col, color='red')
            ax2.tick_params(axis='y', labelcolor='red')

            # Add title and grid
            plt.title(f'Time Series: {score_col} vs {market_col}')
            fig.tight_layout()  # Adjust layout to prevent overlap

            # Show the plot
            plt.show()

def calculate_score():
    df = pd.read_excel("tweets_with_sentiment_scores.xlsx")
    relevance_score_df = pd.read_csv("relevance_score.csv")
    df['calculated_pro_trump_score'] = (df['pro_trump_score'] - df['pro_harris_score']) \
                /(df['pro_trump_score'] + df['pro_harris_score']) * (1 - df['neutral_score_pos']) * relevance_score_df['Relevance Score']
    df['calculated_pro_harris_score'] = (df['pro_harris_score'] - df['pro_trump_score']) \
                /(df['pro_harris_score'] + df['pro_trump_score']) * (1 - df['neutral_score_pos']) * relevance_score_df['Relevance Score']
    df['abs_pro_trump_score'] = (df['pro_trump_score'] - df['pro_harris_score']) * relevance_score_df['Relevance Score']
    df['abs_pro_harris_score'] = (df['pro_harris_score'] - df['pro_trump_score']) * relevance_score_df['Relevance Score']

    df['calculated_neg_trump_score'] = (df['neg_trump_score'] - df['neg_harris_score']) \
                /(df['neg_trump_score'] + df['neg_harris_score']) * (1 - df['neutral_score_neg']) * relevance_score_df['Relevance Score']
    df['calculated_neg_harris_score'] = (df['neg_harris_score'] - df['neg_trump_score']) \
                /(df['neg_harris_score'] + df['neg_trump_score']) * (1 - df['neg_trump_score']) * relevance_score_df['Relevance Score']
    df['abs_neg_trump_score'] = (df['neg_trump_score'] - df['neg_harris_score']) * relevance_score_df['Relevance Score']
    df['abs_neg_harris_score'] = (df['neg_harris_score'] - df['neg_trump_score']) * relevance_score_df['Relevance Score']

    df.to_excel("updated_tweets_with_sentiment_scores.xlsx", index = False)

if __name__ == "__main__":
    calculate_score()
    plot_abs_correl_heatmap()
    plot_abs_time_series()