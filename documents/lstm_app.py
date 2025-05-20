import streamlit as st
import re
import pickle
import numpy as np
import praw
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="wide")

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# ---------------------------
# ⚙️ Load Model & Tokenizer
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("toxic_comments_lstm.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Label categories
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# ---------------------------
# 🧹 Preprocessing Function
# ---------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def prepare_text(text, tokenizer, max_len=100):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded

# ---------------------------
# 🔍 Classify a Single Comment
# ---------------------------
def classify_comment(comment):
    padded = prepare_text(comment, tokenizer)
    predictions = model.predict(padded)[0]
    binary = (predictions > 0.5).astype(int)

    results = []
    for i, label in enumerate(label_cols):
        result = {
            "label": label,
            "binary": binary[i],
            "prob": float(predictions[i])
        }
        results.append(result)
    return results

# ---------------------------
# 🌐 Reddit API Config
# ---------------------------
reddit = praw.Reddit(
    client_id="ki6Z4PMLOTbsm3RgfBLiBQ",
    client_secret="1eaghLUyeWvn-vxUYiEbisJHq3ZXoQ",
    user_agent="nlp_scraper_v1"
)

# Scrape subreddit
def get_comments(subreddit_name, post_limit=10):
    comments_data = []
    for submission in reddit.subreddit(subreddit_name).new(limit=post_limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            if comment.body:
                author = comment.author.name if comment.author else "Unknown"
                comments_data.append((author, comment.body))
    return comments_data

# Detect toxic comments in Reddit data
def detect_toxic_comments(comments):
    toxic_comments = []
    for author, text in comments:
        result = classify_comment(text)
        toxic_labels = [r["label"] for r in result if r["binary"] == 1]
        if toxic_labels:
            toxic_comments.append({
                "author": author,
                "text": text,
                "labels": toxic_labels,
                "probs": result
            })
    return toxic_comments

# ---------------------------
# 🎨 Streamlit UI
# ---------------------------
st.title("💬 Toxic Comment Classifier")

tabs = st.tabs(["📝 Analyze Comment", "🔎 Scan Reddit"])

# ---------------------------
# 📝 Tab 1: Analyze Comment
# ---------------------------
with tabs[0]:
    st.header("Classify a Single Comment")
    user_comment = st.text_area("Enter a comment to analyze:")

    if st.button("Classify Comment"):
        if not user_comment.strip():
            st.warning("Please enter a comment.")
        else:
            result = classify_comment(user_comment)
            st.subheader("📊 Results")
            for r in result:
                status = "🟥 Yes" if r["binary"] == 1 else "✅ No"
                st.write(f"**{r['label']}**: {status} (Probability: {r['prob']:.4f})")

# ---------------------------
# 🔎 Tab 2: Reddit Scanner
# ---------------------------
with tabs[1]:
    st.header("Scan Subreddit for Toxic Comments")
    subreddit_name = st.text_input("Subreddit Name (e.g. askreddit , conspiracy)")
    post_limit = st.slider("Number of posts to scan", 1, 20, 5)

    if st.button("Scan Subreddit"):
        if not subreddit_name.strip():
            st.warning("Please enter a valid subreddit.")
        else:
            with st.spinner("🔍 Scanning comments..."):
                comments = get_comments(subreddit_name, post_limit)
                toxic_found = detect_toxic_comments(comments)

            if toxic_found:
                st.error(f"⚠ Found {len(toxic_found)} toxic comment(s)!")
                for item in toxic_found:
                    st.markdown(f"**👤 Author**: `{item['author']}`")
                    st.markdown(f"**💬 Comment**: {item['text']}")
                    st.markdown(f"**🚨 Toxic Labels**: `{', '.join(item['labels'])}`")
                    st.markdown("---")
            else:
                st.success("✅ No toxic comments found.")

