import os
import re
import joblib
import pandas as pd

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==================================
# SETUP
# ==================================

os.makedirs("models", exist_ok=True)

# ==================================
# LOAD DATA
# ==================================

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)

df = df[['v1', 'v2']]

df.columns = [
    'label',
    'message'
]

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# ==================================
# CLEANING
# ==================================

stop_words = set(
    stopwords.words('english')
)

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['clean_message'] = df[
    'message'
].apply(clean_text)

# ==================================
# TFIDF
# ==================================

tfidf = TfidfVectorizer(
    max_features=5000
)

X = tfidf.fit_transform(
    df['clean_message']
)

y = df['label']

# ==================================
# SPLIT
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================
# MODELS
# ==================================

models = {

    "Naive Bayes":
    MultinomialNB(),

    "Logistic Regression":
    LogisticRegression(
        max_iter=1000
    ),

    "KNN":
    KNeighborsClassifier(),

    "Decision Tree":
    DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest":
    RandomForestClassifier(
        random_state=42
    ),

    "AdaBoost":
    AdaBoostClassifier(
        random_state=42
    ),

    "Gradient Boosting":
    GradientBoostingClassifier(
        random_state=42
    ),

    "XGBoost":
    XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

# ==================================
# TRAINING
# ==================================

results = []

print("\nTraining Models...\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    filename = (
        name.replace(" ", "_")
        .lower()
    )

    joblib.dump(
        model,
        f"models/{filename}.pkl"
    )

    results.append({

        "Model": name,
        "Accuracy": round(
            accuracy,
            4
        ),

        "Precision": round(
            precision,
            4
        ),

        "Recall": round(
            recall,
            4
        ),

        "F1 Score": round(
            f1,
            4
        )

    })

# ==================================
# SAVE TFIDF
# ==================================

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

# ==================================
# LEADERBOARD
# ==================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

results_df.to_csv(
    "models/leaderboard.csv",
    index=False
)

print("\n")
print("=" * 70)
print("MODEL LEADERBOARD")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

best_model_name = results_df.iloc[
    0
]["Model"]

filename = (
    best_model_name
    .replace(" ", "_")
    .lower()
)

best_model = joblib.load(
    f"models/{filename}.pkl"
)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\n")
print(
    f"🏆 Best Performing Model: {best_model_name}"
)

print(
    "\nAll Models Saved Successfully!"
)