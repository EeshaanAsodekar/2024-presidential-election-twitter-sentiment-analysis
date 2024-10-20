import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reference_tweets import pro_trump_tweets, negative_trump_tweets, pro_harris_tweets, negative_harris_tweets, neutral_tweets

def create_labelled_dataset(pro_trump_tweets:list,
                            pro_harris_tweets:list,
                            neutral_tweets: list)->pd.DataFrame:
    '''
    Returns a labelled tweets dataset dataframe
    Columns of the dataset:
        Tweet (the tweet text)
        Label (pro_trump or pro_harris or neutral)
    '''
    # Create labeled DataFrames
    df_pro_trump = pd.DataFrame(pro_trump_tweets, columns=["Tweet"])
    df_pro_trump["Label"] = "pro_trump"

    df_pro_harris = pd.DataFrame(pro_harris_tweets, columns=["Tweet"])
    df_pro_harris["Label"] = "pro_harris"

    df_neutral = pd.DataFrame(neutral_tweets, columns=["Tweet"])
    df_neutral["Label"] = "neutral"

    # Concatenate the DataFrames into a single DataFrame
    df = pd.concat([df_pro_trump, df_pro_harris, df_neutral], ignore_index=True)

    # Shuffle the DataFrame (optional)
    df = df.sample(frac=1).reset_index(drop=True)

    # Display the first few rows
    print(df.head())

    return df

def create_labelled_dataset_negative(negative_trump_tweets:list,
                                     negative_harris_tweets:list,
                                     neutral_tweets: list)->pd.DataFrame:
    '''
    Returns a labelled tweets dataset dataframe for negative sentiments
    Columns of the dataset:
        Tweet (the tweet text)
        Label (neg_trump or neg_harris or neutral)
    '''
    # Create labeled DataFrames
    df_neg_trump = pd.DataFrame(negative_trump_tweets, columns=["Tweet"])
    df_neg_trump["Label"] = "neg_trump"

    df_neg_harris = pd.DataFrame(negative_harris_tweets, columns=["Tweet"])
    df_neg_harris["Label"] = "neg_harris"

    df_neutral = pd.DataFrame(neutral_tweets, columns=["Tweet"])
    df_neutral["Label"] = "neutral"

    # Concatenate the DataFrames into a single DataFrame
    df = pd.concat([df_neg_trump, df_neg_harris, df_neutral], ignore_index=True)

    # Shuffle the DataFrame (optional)
    df = df.sample(frac=1).reset_index(drop=True)

    # Display the first few rows
    print(df.head())

    return df

import re
import string
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean and preprocess tweets
def preprocess_tweet(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # 3. Remove user @mentions but retain the text (e.g., '@JoeBiden' -> 'JoeBiden')
    text = re.sub(r'@\w+', lambda m: m.group(0)[1:], text)

    # 4. Remove hashtags but retain the text (e.g., '#Election2024' -> 'Election2024')
    text = re.sub(r'#(\w+)', r'\1', text)

    # 5. Replace emojis with their text description (using `emoji` library)
    text = emoji.demojize(text, delimiters=(' ', ' '))

    # 6. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 7. Remove numbers (optional, depending on your use case)
    text = re.sub(r'\d+', '', text)

    # 8. Tokenize the text
    words = word_tokenize(text)

    # 9. Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # 10. Join the words back into a single string
    return ' '.join(words)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# **Step 3: Tokenize the Data**
def tokenize_data(examples, tokenizer):
    """Tokenize tweets with the RoBERTa tokenizer."""
    return tokenizer(
        examples["Tweet"],
        padding="max_length",
        truncation=True,
        max_length=512  # Updated from 128 to 512
    )

# **Step 4: Train the RoBERTa Model**
def train_model(labelled_df):
    # Map labels to integers
    label_mapping = {"pro_trump": 0, "pro_harris": 1, "neutral": 2}
    labelled_df["label"] = labelled_df["Label"].map(label_mapping)

    # Initialize the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    dataset = Dataset.from_pandas(labelled_df[["Tweet", "label"]])

    # Use the updated tokenize_data function with max_length=512
    encoded_dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    train_test_split = encoded_dataset.train_test_split(test_size=0.2)

    # Initialize the RoBERTa model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment", num_labels=3
    )

    # Set up training arguments (same as before)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
    )

    # Initialize Trainer (same as before)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./results")
    tokenizer.save_pretrained("./results")

    return model

def train_negative_model(labelled_df):
    # Map labels to integers
    label_mapping = {"neg_trump": 0, "neg_harris": 1, "neutral": 2}
    labelled_df["label"] = labelled_df["Label"].map(label_mapping)

    # Initialize the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    dataset = Dataset.from_pandas(labelled_df[["Tweet", "label"]])

    # Use the updated tokenize_data function with max_length=512
    encoded_dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    train_test_split = encoded_dataset.train_test_split(test_size=0.2)

    # Initialize the RoBERTa model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment", num_labels=3
    )

    # Set up training arguments (same as before)
    training_args = TrainingArguments(
        output_dir="./negative_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./negative_logs",
        logging_steps=10,
        save_total_limit=1,
    )

    # Initialize Trainer (same as before)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained("./negative_results")
    tokenizer.save_pretrained("./negative_results")

    return model


def generate_sentiment_scores(model, df, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("./results")

    pro_trump_scores = []
    pro_harris_scores = []
    neutral_scores = []

    # Process data in batches
    for i in range(0, len(df), batch_size):
        batch_texts = df["cleaned_text"][i:i+batch_size].tolist()
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            encodings = {key: val.cuda() for key, val in encodings.items()}
            model = model.cuda()

        # Predict with the model
        with torch.no_grad():
            outputs = model(**encodings)

        # Get probabilities from logits
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Append batch results to the lists
        pro_trump_scores.extend(probs[:, 0])
        pro_harris_scores.extend(probs[:, 1])
        neutral_scores.extend(probs[:, 2])

    # Assign scores to DataFrame
    df["pro_trump_score"] = pro_trump_scores
    df["pro_harris_score"] = pro_harris_scores
    df["neutral_score_pos"] = neutral_scores

    return df



def generate_negative_sentiment_scores(model, df, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained("./negative_results")

    neg_trump_scores = []
    neg_harris_scores = []
    neutral_scores = []

    # Process data in batches
    for i in range(0, len(df), batch_size):
        batch_texts = df["cleaned_text"][i:i+batch_size].tolist()
        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move tensors to GPU if available
        if torch.cuda.is_available():
            encodings = {key: val.cuda() for key, val in encodings.items()}
            model = model.cuda()

        # Predict with the model
        with torch.no_grad():
            outputs = model(**encodings)

        # Get probabilities from logits
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Append batch results to the lists
        neg_trump_scores.extend(probs[:, 0])
        neg_harris_scores.extend(probs[:, 1])
        neutral_scores.extend(probs[:, 2])

    # Assign scores to DataFrame
    df["neg_trump_score"] = neg_trump_scores
    df["neg_harris_score"] = neg_harris_scores
    df["neutral_score_neg"] = neutral_scores

    return df



def combined_scoring():
    # Create the positive labelled dataset
    labelled_df = create_labelled_dataset(pro_trump_tweets, pro_harris_tweets, neutral_tweets)

    # Create the negative labelled dataset
    negative_labelled_df = create_labelled_dataset_negative(negative_trump_tweets, negative_harris_tweets, neutral_tweets)

    # Load and clean your main dataset
    # tweets_df = pd.read_csv(r'data\raw\tweets_Presidential_Election_data_Oct15_2024.csv')
    tweets_df = pd.read_excel(r'data\raw\tweets_combined.xlsx')
    tweets_df["Text"] = tweets_df["Text"].astype(str)
    tweets_df["cleaned_text"] = tweets_df["Text"].apply(preprocess_tweet)

    print(tweets_df.head())
    print(labelled_df.head())
    print(negative_labelled_df.head())

    # Train the positive sentiment RoBERTa model
    model = train_model(labelled_df)

    # Generate positive sentiment scores for the main dataset
    tweets_df_with_scores = generate_sentiment_scores(model, tweets_df)

    # Train the negative sentiment RoBERTa model
    negative_model = train_negative_model(negative_labelled_df)

    # Generate negative sentiment scores for the main dataset
    tweets_df_with_scores = generate_negative_sentiment_scores(negative_model, tweets_df_with_scores)

    # Save the results to an Excel file
    tweets_df_with_scores.to_excel("tweets_with_sentiment_scores.xlsx", index=False)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')  # Download VADER lexicon if not already downloaded

def individual_scoring():
    tweets_df = pd.read_excel('data/raw/tweets_classified.xlsx')
    # print(tweets_df.value_counts())
    # print(tweets_df.columns)
    print(tweets_df['Label'].value_counts())

    # Assuming 'Label' column contains labels 'trump', 'kamala', 'neutral'
    # Split the DataFrame into Trump and Kamala DataFrames
    trump_df = tweets_df[tweets_df['Label'] == 'trump'].copy()
    kamala_df = tweets_df[tweets_df['Label'] == 'harris'].copy()

    # Initialize VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Function to get sentiment score
    def get_sentiment_score(text):
        return sid.polarity_scores(text)['compound']

    # Apply sentiment analysis to the tweets
    trump_df['sentiment'] = trump_df['Text'].apply(get_sentiment_score)
    kamala_df['sentiment'] = kamala_df['Text'].apply(get_sentiment_score)

    # Return or save the dataframes as needed
    # For example, print the first few rows
    print(trump_df.head())
    trump_df.to_csv("vader_trump_score.csv")
    print(kamala_df.head())
    kamala_df.to_csv("vader_harris_score.csv")


if __name__ == "__main__":
    combined_scoring()
    # individual_scoring()