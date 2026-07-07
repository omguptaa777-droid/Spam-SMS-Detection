import streamlit as st
import pickle
import joblib
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


def transform_text(msg):
    msg = msg.lower()  # lowercase
    msg = word_tokenize(msg)  # tokenization

    words = []
    stop_words = set(stopwords.words('english'))

    for w in msg:
        # remove punctuations and stopwords
        if w.isalnum() and w not in stop_words:
            words.append(w)

    msg = words[:]
    words.clear()

    for w in msg:
        # stemming
        words.append(ps.stem(w))

    return ' '.join(words)

tfidf = pickle.load(open('vectorizer2.pkl', 'rb'))
model = joblib.load("model2.joblib")

st.title("📩 Spam SMS Detector")

st.markdown("""
Detect whether an SMS is **Spam** or **Ham** using a Machine Learning model trained on thousands of real SMS messages.

Type a message below and click **Predict**.
""")

with st.container(border=True):
    input_msg = st.text_area(
        "Enter SMS Message",
        height=180,
        placeholder="Example:\nCongratulations! You have won ₹50,000. Click here to claim."
    )

with st.sidebar:

    st.header("About")

    st.write("""
This application uses:

- TF-IDF Vectorizer
- Machine Learning Classifier
- NLTK preprocessing
- Streamlit
    """)

    st.divider()

    st.write("Developed by Om Gupta")

if st.button("🔍 Predict"):
    with st.spinner("Analyzing message..."):
        transformed_text = transform_text(input_msg)
        vector = tfidf.transform([transformed_text])
        prediction = model.predict(vector)[0]
    if prediction == 0:
        st.success("✅ Legitimate Message")
    else:
        st.error("🚨 Spam Message Detected")

    st.subheader("Message Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Words", len(input_msg.split()))
    col2.metric("Characters", len(input_msg))
    col3.metric("Processed Words", len(transformed_text.split()))

    prob = model.predict_proba(vector)[0]
    spam_prob = prob[1]
    st.progress(float(spam_prob))
    st.write(f"Spam Probability: {spam_prob * 100:.2f}%")


st.divider()

st.caption("Made with ❤️ using Streamlit and Scikit-Learn")