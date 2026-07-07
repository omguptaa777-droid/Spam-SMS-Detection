# 📩 Spam SMS Classifier

A Machine Learning-based web application that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and an ensemble learning approach.

The project follows a complete end-to-end machine learning pipeline, starting from data analysis and preprocessing to model selection, evaluation, and deployment using Streamlit.

---

## 🚀 Demo

> https://spam-sms-detection-5a5h7cv7qzkv2yzoxvhmau.streamlit.app/

---

## 📌 Project Overview

Spam SMS detection is a binary text classification problem where the objective is to accurately identify unsolicited messages while minimizing false alarms for legitimate messages.

This project was built by following a systematic machine learning workflow rather than relying on a single model. Multiple algorithms and ensemble techniques were evaluated before selecting the final model based on comprehensive performance metrics.

---

## 📂 Dataset

To improve the robustness of the classifier, two publicly available SMS datasets were combined.

### Initial Challenge

The original SMS Spam Collection dataset was **highly imbalanced**, containing significantly more legitimate (Ham) messages than Spam messages.

Although the initial model achieved high overall accuracy, it struggled to correctly classify **short spam messages**, often predicting them as legitimate because of the limited representation of such examples in the training data.

### Solution

To address this issue:

* Collected an additional SMS spam dataset.
* Merged both datasets into a single dataset.
* Removed duplicate entries.
* Re-trained the models using the merged dataset.

### Final Data Distribution

| Class | Percentage |
| ----- | ---------: |
| Ham   |    **74%** |
| Spam  |    **26%** |

This improved the representation of spam messages, particularly short spam texts, resulting in a significantly more reliable classifier.

---

## 📊 Exploratory Data Analysis (EDA)

A detailed exploratory analysis was performed before model development.

The analysis included:

* Class distribution analysis
* Message length distribution
* Word count analysis
* Character count analysis
* Comparison between Spam and Ham messages
* Most frequent words
* Most frequent spam keywords
* Text preprocessing insights
* Duplicate message detection
* Correlation between message statistics

The insights obtained from EDA helped guide preprocessing decisions and improve model performance.

---

## ⚙️ Text Preprocessing

The following preprocessing pipeline was applied:

* Convert text to lowercase
* Tokenization using NLTK
* Removal of punctuation
* Removal of stopwords
* Stemming using Porter Stemmer

The processed text was then converted into numerical feature vectors using:

**TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization**

---

## 🤖 Models Evaluated

The following machine learning algorithms were trained and evaluated:

* Naive Bayes
* Support Vector Classifier (SVC)
* K-Nearest Neighbors (KNN)
* Random Forest
* Extra Trees Classifier
* Bagging Classifier

After evaluating the individual models, ensemble learning techniques were explored:

* Voting Classifier
* Stacking Classifier

---

## 📈 Model Performance

### Individual Models

| Model         | Accuracy |  Precision |
| ------------- | -------: | ---------: |
| SVC           |   96.53% |     96.16% |
| Extra Trees   |   96.10% |     95.36% |
| Random Forest |   95.82% | **97.70%** |
| Bagging       |   95.84% |     95.09% |
| Naive Bayes   |   93.49% |     88.28% |
| KNN           |   87.98% |     96.15% |

---

### Ensemble Models

| Model                 |   Accuracy |  Precision |
| --------------------- | ---------: | ---------: |
| **Voting Classifier** | **96.78%** | **96.50%** |
| Stacking Classifier   |     95.96% |     93.04% |

---

## 📋 Classification Report

### Voting Classifier

| Class | Precision |   Recall | F1-score |
| ----- | --------: | -------: | -------: |
| Ham   |      0.97 |     0.99 |     0.98 |
| Spam  |  **0.96** | **0.91** | **0.94** |

**Overall Accuracy:** **96.78%**

The Voting Classifier achieved the best balance between precision and recall while maintaining the highest overall accuracy and F1-score.

---

## ✅ Final Model Selection

Several models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Complete Classification Report

Although the Random Forest model achieved the highest spam precision, its recall was comparatively lower, causing more spam messages to be missed.

The Voting Classifier demonstrated the best overall balance by:

* Achieving the highest overall accuracy
* Maintaining high precision
* Improving spam recall
* Producing the highest F1-score among the evaluated models

Based on these observations, the **Voting Classifier** was selected as the final model for deployment.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* NLTK
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

## 📁 Project Structure

```text
Spam-SMS-Classifier/
│
├── app.py
├── train_model.ipynb
├── vectorizer.joblib
├── requirements.txt
├── README.md
├── data/
├── assets/
└── .gitignore
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/spam-sms-classifier.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the trained model from the Hugging Face repository and place it in the project root.

Run the application:

```bash
streamlit run app.py
```

---

## 📦 Model

The trained model is hosted separately on Hugging Face because it exceeds GitHub's file size limit.

> https://huggingface.co/Om-Gupta/spam-sms-classifier-model

---

## 🎯 Key Outcomes

* Successfully addressed class imbalance by augmenting the training data with an additional dataset.
* Improved the classifier's ability to detect **short spam messages**, which were frequently misclassified in the initial version.
* Built a complete end-to-end NLP pipeline including EDA, preprocessing, feature engineering, model comparison, ensemble learning, and deployment.
* Selected the final model based on a comprehensive evaluation using multiple performance metrics rather than relying solely on accuracy.

---

## 👨‍💻 Author

**Om Gupta**

Undergraduate Student | National Institute of Technology Bhopal

Interested in Machine Learning, Natural Language Processing, and Data Science.
