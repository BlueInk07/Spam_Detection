import pandas as pd
import re
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

stop_words = set(stopwords.words('english'))

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['clean_message'] = df['message'].apply(clean_text)

print(df[['message', 'clean_message']].head())