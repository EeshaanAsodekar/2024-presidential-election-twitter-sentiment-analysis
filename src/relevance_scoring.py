import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# Load your tweet dataset
df = pd.read_csv(r'data\raw\tweets_Presidential_Election_data_Oct15_2024.csv')

# Ensure 'Text' column is of string type
df['Text'] = df['Text'].astype(str)

# Define the weighted keyword dictionary
weighted_keywords = {
    # High Weight (3 points)
    'trump': 3, 'donald trump': 3, 'kamala': 3, 'kamala harris': 3, 'harris': 3,
    'president': 3, 'presidency': 3, 'election': 3, 'vote': 3, 'voting': 3,
    'ballot': 3, 'campaign': 3, 'rally': 3, 'debate': 3, 'maga': 3, '2024': 3,
    'trump 2024': 3, 'harris walz': 3, 'tim walz': 3, 'vice president': 3,
    # Medium Weight (2 points)
    'biden': 2, 'joe biden': 2, 'walz': 2, 'democrat': 2, 'democrats': 2,
    'republican': 2, 'republicans': 2, 'gop': 2, 'polls': 2, 'swing state': 2,
    'candidate': 2, 'nominate': 2, 'nomination': 2, 'leftists': 2, 'conservatives': 2,
    'liberal': 2, 'progressives': 2, 'obama': 2, 'pence': 2, 'hillary': 2, 'clinton': 2,
    'nikki haley': 2, 'election day': 2, 'voter': 2, 'voters': 2,
    # Low Weight (1 point)
    'support': 1, 'proud': 1, 'amazing': 1, 'effective': 1, 'strong': 1,
    'win': 1, 'winning': 1, 'best': 1, 'lead': 1, 'leading': 1, 'momentum': 1,
    'hate': 1, 'disgusted': 1, 'liar': 1, 'lies': 1, 'corrupt': 1,
    'traitor': 1, 'worst': 1, 'dangerous': 1, 'scared': 1, 'stupid': 1,
    'crazy': 1, 'weak': 1, 'loser': 1, 'rambling': 1, 'incoherent': 1,
    'insane': 1, 'idiot': 1, 'deranged': 1, 'economy': 1, 'jobs': 1,
    'manufacturing': 1, 'inflation': 1, 'healthcare': 1, 'covid': 1,
    'pandemic': 1, 'abortion': 1, 'immigration': 1, 'border': 1,
    'climate': 1, 'gun control': 1, 'education': 1, 'foreign policy': 1,
    'fema': 1, 'hurricane': 1, 'military': 1, 'putin': 1, 'russia': 1,
    'china': 1, 'north korea': 1, 'iran': 1, 'afghanistan': 1, 'cia': 1,
    'fbi': 1, 'classified documents': 1, 'elon musk': 1, 'cnn': 1, 'fox news': 1,
    'town hall': 1, 'speech': 1, 'interview': 1, 'press conference': 1,
    'univision': 1, 'debates': 1, 'election fraud': 1, 'voter suppression': 1,
    'polling': 1, 'mail-in ballot': 1, 'absentee ballot': 1,
}

# Function to calculate relevance score
def calculate_relevance_score(tweet):
    tweet_lower = tweet.lower()
    total_words = len(re.findall(r'\w+', tweet_lower))
    keyword_points = 0

    for keyword, weight in weighted_keywords.items():
        # Escape special characters in keywords for regex
        escaped_keyword = re.escape(keyword)
        # Use word boundaries to match whole words or phrases
        pattern = r'\b' + escaped_keyword + r'\b'
        matches = re.findall(pattern, tweet_lower)
        keyword_points += len(matches) * weight

    relevance_score = keyword_points / total_words if total_words > 0 else 0
    return relevance_score

# Apply the function to the DataFrame
df['Relevance Score'] = df['Text'].apply(calculate_relevance_score)

# Optionally, normalize the relevance scores between 0 and 1
scaler = MinMaxScaler()
df['Relevance Score Normalized'] = scaler.fit_transform(df[['Relevance Score']])

# Display the results
print(df[['Text', 'Relevance Score', 'Relevance Score Normalized']])
df[['Text', 'Relevance Score', 'Relevance Score Normalized']].to_excel('rel_score_word_dict_approach.xlsx')

# TODO: add to relevancce score based on the hastags ... and @'s .... 
# keep 90% weight on the tweet text ... and 10% on the hastags and @s 