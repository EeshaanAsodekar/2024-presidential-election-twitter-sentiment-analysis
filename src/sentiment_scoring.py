import pandas as pd

pro_trump_tweets = [
    "I cast my ballot today for Donald J Trump.",
    "My 24 y/o son is casting his first ever presidential vote for Donald J Trump. I couldn’t be prouder.",
    "I would like to nominate myself as the deportation czar in President Donald Trump's second term.",
    "Hurricane Milton is a Category 5. I’m still voting for Trump. Come hell or high water, I’ll be voting.",
    "I went bankrupt with Biden/Harris. I won Super Bowl with Trump. My story resembles so many others. Striving with Trump. Broke with Biden/Harris.",
    "I’m a centrist independent moderate who has voted 3rd party extensively in the past, because of corruption in the major parties. This year I’m voting for Donald Trump because he’s the most independent candidate running.",
    "If Trump gets elected, America as we know it will no longer exist. The most patriotic thing ALL people can do is to vote for #KamalaHarris.",  # Note: This tweet appears pro-Harris but may contain a typo.
    "Donald Trump donated $25 million to hurricane victims and he’s now hosting 275 linemen in Florida at his resort for free.",
    "Trump supporters WANT voter ID and proof of citizenship. Harris supporters don’t. Not sure why we make it so complicated.",
    "Donald Trump is igniting this presidential campaign!"
]

pro_harris_tweets = [
    "Harris keeps overperforming in districts Biden barely won in Pennsylvania. Anyone who says Trump has this election in the bag is delusional.",
    "Kamala Harris raised over a billion dollars after being nominated for president, a record-breaking amount.",
    "Who thinks Kamala will be a way better president than Trump?",
    "Kamala Harris deserves a chance to prove that she can lead America. We already know what Trump has to offer; chaos, lies, and division.",
    "Vote for Kamala Harris and Tim Walz because America won’t survive Trump a second time.",
    "Harris is stealing trainloads of Republican voters from Trump, but Trump isn’t taking any Democrat voters from Harris.",
    "Kamala Harris will fight to restore Roe v. Wade. Let’s get this done.",
    "Kamala Harris stunned the audience into silence with her impassioned reminiscence of what it was like when Trump sent COVID tests to Putin while Americans were dying.",
    "I just spoke to my Uber driver in Detroit, Michigan, and he told me he’s voting for Kamala Harris. He's originally from Gambia and is a registered Independent.",
    "With all these Republicans coming out to endorse Harris, it’s fair to say she’s a 'bipartisan' presidential candidate."
]

neutral_tweets = [
    "I literally have people freaking out & texting me about all these polls. Polls don't vote! And most are bought & paid for. So sick of this.",
    "Again, media is just letting it go that Trump won't release his medical records. If Biden did the same it would be a scandal.",
    "We have 28 days to go! All Gas, No Brakes. Bring your friends & family!",
    "I'm begging everybody, if you can vote early, please do so. Do not wait until election day. As you can see with hurricanes bearing down on some of these states.",
    "Everyone saying that people are leaving Twitter right now, think about this: Twitter has its problems but it is a major source of how we get information out.",
    "Don’t listen to the polls, pundits, media, or news. Listen to your soul.",
    "Polls be like: Is Donald Trump guilty of the crimes he has been accused? Yes: 60%, No: 34%. Is Donald Trump a moral monster? Yes: 74%.",
    "Polls don’t vote.",
    "We have a BIG group coming this weekend to Lancaster County, PA to knock doors for Harris-Walz!",
    "Good morning everyone!!"
]

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


# import pandas as pd
# import torch
# import torch.nn.functional as F
# from transformers import Bert   , BertForSequenceClassification, Trainer, TrainingArguments
# from datasets import Dataset

# # **Step 3: Tokenize the Data**
# def tokenize_data(examples, tokenizer):
#     return tokenizer(examples["Tweet"], padding="max_length", truncation=True, max_length=128)

# # **Step 4: Train the BERT Model**
# def train_model(labelled_df):
#     # Map labels to integers
#     label_mapping = {"pro_trump": 0, "pro_harris": 1, "neutral": 2}
#     labelled_df["label"] = labelled_df["Label"].map(label_mapping)

#     # Initialize the tokenizer and dataset
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     dataset = Dataset.from_pandas(labelled_df[["Tweet", "label"]])
#     encoded_dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
#     train_test_split = encoded_dataset.train_test_split(test_size=0.2)

#     # Initialize the model
#     model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

#     # Set up training arguments
#     training_args = TrainingArguments(
#         output_dir="./results",
#         evaluation_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         logging_dir="./logs",
#         logging_steps=10,
#         save_total_limit=1,
#     )

#     # Initialize Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_test_split["train"],
#         eval_dataset=train_test_split["test"],
#     )

#     # Train the model
#     trainer.train()

#     # Save the model
#     model.save_pretrained("./results")
#     tokenizer.save_pretrained("./results")

#     return model

# # **Step 5: Generate Sentiment Scores**
# def generate_sentiment_scores(model, df):
#     tokenizer = BertTokenizer.from_pretrained("./results")

#     # Tokenize tweets
#     encodings = tokenizer(list(df["cleaned_text"]), padding=True, truncation=True, return_tensors="pt")

#     # Predict with the model
#     with torch.no_grad():
#         outputs = model(**encodings)

#     # Get probabilities
#     probs = F.softmax(outputs.logits, dim=-1)

#     # Assign scores to DataFrame
#     df["pro_trump_score"] = probs[:, 0].numpy()
#     df["pro_harris_score"] = probs[:, 1].numpy()
#     df["neutral_score"] = probs[:, 2].numpy()

#     return df

# # **Step 6: Main Function**
# if __name__ == "__main__":
#     # Create the labelled dataset
#     labelled_df = create_labelled_dataset(pro_trump_tweets, pro_harris_tweets, neutral_tweets)

#     # Load and clean your main dataset
#     # tweets_df = pd.read_csv("data/raw/tweets_Presidential_Election_data_Oct15_2024.csv")
#     tweets_df = pd.read_csv("data/raw/subset.csv")
#     tweets_df["Text"] = tweets_df["Text"].astype(str)
#     tweets_df["cleaned_text"] = tweets_df["Text"].apply(preprocess_tweet)

#     print(tweets_df.head())
#     print(labelled_df.head())

#     # Train the model
#     model = train_model(labelled_df)

#     # Generate sentiment scores for the main dataset
#     tweets_df_with_scores = generate_sentiment_scores(model, tweets_df)

#     # Save the results
#     tweets_df_with_scores.to_excel("tweets_with_sentiment_scores.xlsx", index=False)
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# **Step 3: Tokenize the Data**
def tokenize_data(examples, tokenizer):
    """Tokenize tweets with the RoBERTa tokenizer."""
    return tokenizer(examples["Tweet"], padding="max_length", truncation=True, max_length=128)

# **Step 4: Train the RoBERTa Model**
def train_model(labelled_df):
    # Map labels to integers
    label_mapping = {"pro_trump": 0, "pro_harris": 1, "neutral": 2}
    labelled_df["label"] = labelled_df["Label"].map(label_mapping)

    # Initialize the tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    dataset = Dataset.from_pandas(labelled_df[["Tweet", "label"]])
    encoded_dataset = dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    train_test_split = encoded_dataset.train_test_split(test_size=0.2)

    # Initialize the RoBERTa model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment", num_labels=3
    )

    # Set up training arguments
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

    # Initialize Trainer
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

# **Step 5: Generate Sentiment Scores**
def generate_sentiment_scores(model, df):
    tokenizer = AutoTokenizer.from_pretrained("./results")

    # Tokenize tweets
    encodings = tokenizer(list(df["cleaned_text"]), padding=True, truncation=True, return_tensors="pt")

    # Predict with the model
    with torch.no_grad():
        outputs = model(**encodings)

    # Get probabilities from logits
    probs = F.softmax(outputs.logits, dim=-1)

    # Assign scores to DataFrame
    df["pro_trump_score"] = probs[:, 0].numpy()
    df["pro_harris_score"] = probs[:, 1].numpy()
    df["neutral_score"] = probs[:, 2].numpy()

    return df

# **Step 6: Main Function**
if __name__ == "__main__":
    # Create the labelled dataset
    labelled_df = create_labelled_dataset(pro_trump_tweets, pro_harris_tweets, neutral_tweets)

    # Load and clean your main dataset
    tweets_df = pd.read_csv("data/raw/subset.csv")
    tweets_df["Text"] = tweets_df["Text"].astype(str)
    tweets_df["cleaned_text"] = tweets_df["Text"].apply(preprocess_tweet)

    print(tweets_df.head())
    print(labelled_df.head())

    # Train the RoBERTa model
    model = train_model(labelled_df)

    # Generate sentiment scores for the main dataset
    tweets_df_with_scores = generate_sentiment_scores(model, tweets_df)

    # Save the results to an Excel file
    tweets_df_with_scores.to_excel("tweets_with_sentiment_scores.xlsx", index=False)
