# Spam_Detection<div align="center">

# 🛡️ SpamShield AI Arena

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=24&duration=3000&pause=1000&color=3B82F6&center=true&vCenter=true&width=800&lines=NLP+Powered+Spam+Detection;8+Machine+Learning+Models;Weighted+Ensemble+Voting;Streamlit+Interactive+Dashboard" />

<br>

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge\&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-blue?style=for-the-badge\&logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-blue?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-blue?style=for-the-badge\&logo=streamlit)

</div>

---

## 🚀 Overview

SpamShield AI Arena is a Machine Learning powered spam detection system that classifies SMS messages as **Spam** or **Ham (Legitimate Messages)**.

Unlike traditional spam detectors that rely on a single model, SpamShield AI Arena compares predictions from **8 different Machine Learning algorithms** and uses a **weighted voting mechanism** based on model performance.

This project demonstrates:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Model Training & Evaluation
* Ensemble Decision Making
* Streamlit Deployment

---

## 🎯 Key Features

✅ Spam vs Ham Classification

✅ TF-IDF Text Vectorization

✅ 8 Machine Learning Models

✅ Weighted Voting System

✅ Interactive Streamlit Dashboard

✅ Model Performance Leaderboard

✅ Confidence Score Calculation

✅ Real-Time Prediction

---

## 🧠 Machine Learning Models Used

| Model               | Type              |
| ------------------- | ----------------- |
| Naive Bayes         | Probabilistic     |
| Logistic Regression | Linear Classifier |
| K-Nearest Neighbors | Distance Based    |
| Decision Tree       | Tree Based        |
| Random Forest       | Ensemble          |
| AdaBoost            | Boosting          |
| Gradient Boosting   | Boosting          |
| XGBoost             | Advanced Boosting |

---

## ⚙️ Project Pipeline

```text
SMS Message
     │
     ▼
Text Cleaning
     │
     ▼
TF-IDF Vectorization
     │
     ▼
8 Machine Learning Models
     │
     ▼
Weighted Voting
     │
     ▼
Final Prediction
```

---

## 📊 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

### Best Performing Model

🏆 XGBoost

```text
Accuracy  : 97.58%
Precision : 99.19%
Recall    : 82.55%
F1 Score  : 90.11%
```

---

## 📈 Why F1 Score?

Spam datasets are imbalanced.

```text
Ham  : 4825
Spam : 747
```

Using Accuracy alone can be misleading.

Therefore models were ranked using **F1 Score**, which balances:

* Precision
* Recall

---

## 🛡️ Weighted Voting System

Instead of giving every model equal influence:

```text
KNN Vote = XGBoost Vote
```

Weighted voting uses model performance:

```text
Higher F1 Score
        ↓
Higher Voting Power
```

This improves prediction reliability.

---

## 🖥️ Application Preview

### Arena Tab

Displays:

* Final Verdict
* Confidence Score
* Individual Model Predictions

### Leaderboard Tab

Displays:

* Model Rankings
* F1 Score Comparison
* Performance Analysis

---

## 📂 Project Structure

```text
Spam_Detection/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── spam.csv
│
├── models/
│   └── leaderboard.csv
│
├── src/
│   ├── explore_data.py
│   ├── preprocess.py
│   ├── train_models.py
│   └── predict.py
│
└── notebooks/
```

---

## 🛠️ Installation

Clone repository

```bash
git clone https://github.com/BlueInk07/Spam_Detection.git
```

Move into project

```bash
cd Spam_Detection
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
streamlit run app.py
```

---

## 💻 Tech Stack

### Languages

* Python

### Libraries

* Pandas
* NumPy
* NLTK
* Scikit-Learn
* XGBoost
* Joblib

### Deployment

* Streamlit

---

## 🔮 Future Improvements

* Deep Learning Models (LSTM)
* Transformer Based Models (BERT)
* Explainable AI Integration
* Cloud Deployment
* Real Email Spam Detection

---

## 👨‍💻 Author

### Simran Chouhan

AI/ML Enthusiast • CSE Student • UI/UX Designer

💼 LinkedIn: [www.linkedin.com/in/simran-chouhan17](http://www.linkedin.com/in/simran-chouhan17)

---

<div align="center">

### ⭐ If you found this project interesting, consider starring the repository.

</div>
