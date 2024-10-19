import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reference_tweets import pro_trump_tweets, negative_trump_tweets,pro_harris_tweets,negative_harris_tweets, neutral_tweets

def create_labelled_dataset(trump_tweets:list,
                            harris_tweets:list,
                            neutral_tweets: list)->pd.DataFrame:
    '''
    Returns a labelled tweets dataset dataframe
    Columns of the dataset:
        Tweet (the tweet text)
        Label (trump or harris or neutral)
    '''
    # Create labeled DataFrames
    df_trump = pd.DataFrame(trump_tweets, columns=["Tweet"])
    df_trump["Label"] = "trump"

    df_harris = pd.DataFrame(harris_tweets, columns=["Tweet"])
    df_harris["Label"] = "harris"

    df_neutral = pd.DataFrame(neutral_tweets, columns=["Tweet"])
    df_neutral["Label"] = "neutral"

    # Concatenate the DataFrames into a single DataFrame
    df = pd.concat([df_trump, df_harris, df_neutral], ignore_index=True)

    # Shuffle the DataFrame (optional)
    df = df.sample(frac=1).reset_index(drop=True)

    return df

def classify_tweets(tweets_labelled_df: pd.DataFrame, tweets_df: pd.DataFrame) -> pd.DataFrame:
    # Split labelled dataset into features (X) and labels (y)
    X = tweets_labelled_df['Tweet']
    y = tweets_labelled_df['Label']

    # Encode labels to numeric values for compatibility with XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_vect = vectorizer.fit_transform(X)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y_encoded, test_size=0.2, random_state=42)

    # Initialize models to test
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'XGBoost': XGBClassifier(eval_metric='mlogloss'),
        'SVM': SVC(kernel='linear', probability=True, class_weight='balanced')
    }

    # Train and evaluate each model, storing the one with the best accuracy
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Improved output formatting
        print(f"{model_name:<20} | Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    print(f"\nBest Model: {type(best_model).__name__} with Accuracy: {best_accuracy:.2f}")

    # Use the best model to classify tweets in tweets_df
    tweets_df_vect = vectorizer.transform(tweets_df['Text'])
    predicted_labels_encoded = best_model.predict(tweets_df_vect)

    # Decode the numeric predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)

    # Add the predicted labels as a new column in tweets_df
    tweets_df['Label'] = predicted_labels

    return tweets_df

if __name__ == "__main__":
    # Load tweets dataset
    tweets_df = pd.read_csv("data/raw/tweets_Presidential_Election_data_Oct15_2024.csv")
    tweets_df["Text"] = tweets_df["Text"].astype(str)

    # Create the labelled tweets dataset
    trump_tweet_list = pro_trump_tweets + negative_trump_tweets
    harris_tweets_list = pro_harris_tweets + negative_harris_tweets
    neutral_tweets_list = neutral_tweets

    tweets_labelled_df = create_labelled_dataset(trump_tweet_list, harris_tweets_list, neutral_tweets_list)

    # Classify the tweets
    classified_tweets_df = classify_tweets(tweets_labelled_df, tweets_df)

    # Print results
    print(classified_tweets_df.head())
    print(tweets_labelled_df['Label'].value_counts())
    print(classified_tweets_df['Label'].value_counts())
    subset_classified_tweets_df = classified_tweets_df[['Text','Label']]
    subset_classified_tweets_df.to_excel('data/raw/tweets_classified.xlsx')