import joblib

model = joblib.load(
    "models/best_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

message = input("Enter Message: ")

message_vector = vectorizer.transform(
    [message]
)

prediction = model.predict(
    message_vector
)

if prediction[0] == 1:
    print("\nSPAM")
else:
    print("\nHAM")