import streamlit as st
import pandas as pd
import joblib
import re

from nltk.corpus import stopwords

# ==================================
# PAGE
# ==================================

st.set_page_config(
    page_title="SpamShield AI Arena",
    page_icon="🛡️",
    layout="wide"
)

# ==================================
# CLEANING FUNCTION
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

# ==================================
# LOAD FILES
# ==================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

leaderboard = pd.read_csv(
    "models/leaderboard.csv"
)

models = {

    "Naive Bayes":
    joblib.load(
        "models/naive_bayes.pkl"
    ),

    "Logistic Regression":
    joblib.load(
        "models/logistic_regression.pkl"
    ),

    "KNN":
    joblib.load(
        "models/knn.pkl"
    ),

    "Decision Tree":
    joblib.load(
        "models/decision_tree.pkl"
    ),

    "Random Forest":
    joblib.load(
        "models/random_forest.pkl"
    ),

    "AdaBoost":
    joblib.load(
        "models/adaboost.pkl"
    ),

    "Gradient Boosting":
    joblib.load(
        "models/gradient_boosting.pkl"
    ),

    "XGBoost":
    joblib.load(
        "models/xgboost.pkl"
    )
}

# ==================================
# UI
# ==================================

st.title(
    "🛡️ SpamShield AI Arena"
)

tab1, tab2 = st.tabs([
    "Arena",
    "Leaderboard"
])

# ==================================
# TAB 1
# ==================================

with tab1:

    message = st.text_area(
        "Paste a Message",
        height=180
    )

    if st.button(
        "Analyze Message"
    ):

        if message.strip() == "":

            st.warning(
                "Enter a message."
            )

        else:

            cleaned = clean_text(
                message
            )

            vector = vectorizer.transform(
                [cleaned]
            )

            results = []

            spam_weight = 0
            ham_weight = 0

            for name, model in models.items():

                prediction = model.predict(
                    vector
                )[0]

                model_f1 = float(

                    leaderboard[
                        leaderboard[
                            "Model"
                        ] == name
                    ][
                        "F1 Score"
                    ].iloc[0]

                )

                if prediction == 1:

                    spam_weight += model_f1

                else:

                    ham_weight += model_f1

                results.append({

                    "Model": name,

                    "Prediction":
                    "Spam"
                    if prediction == 1
                    else "Ham",

                    "F1 Weight":
                    round(
                        model_f1,
                        3
                    )

                })

            results_df = pd.DataFrame(
                results
            )

            st.subheader(
                "Final Verdict"
            )

            if spam_weight > ham_weight:

                st.error(
                    "🚨 SPAM DETECTED"
                )

            else:

                st.success(
                    "✅ HAM MESSAGE"
                )

            confidence = (
                max(
                    spam_weight,
                    ham_weight
                )
                /
                (
                    spam_weight +
                    ham_weight
                )
            ) * 100

            st.metric(
                "Confidence Score",
                f"{confidence:.1f}%"
            )

            st.subheader(
                "Model Votes"
            )

            st.dataframe(
                results_df,
                use_container_width=True
            )

# ==================================
# TAB 2
# ==================================

with tab2:

    st.subheader(
        "Model Leaderboard"
    )

    st.dataframe(
        leaderboard,
        use_container_width=True
    )

    st.bar_chart(
        leaderboard.set_index(
            "Model"
        )[
            "F1 Score"
       ]
    )