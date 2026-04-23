import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from deep_translator import GoogleTranslator
import time
import os

port = int(os.environ.get("PORT", 8501))
st.run(host="0.0.0.0", port=port)

# -----------------------
# PAGE CONFIG (FIRST)
# -----------------------
st.set_page_config(page_title="🔥 Ultimate Analyzer", page_icon="🔥")

# -----------------------
# LOAD MODELS (FAST)
# -----------------------
@st.cache_resource
def load_models():
    vader = SentimentIntensityAnalyzer()
    bert = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return vader, bert

vader, bert = load_models()

# -----------------------
# TRANSLATE
# -----------------------
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# -----------------------
# WORD LIST
# -----------------------
pos_words = ["good","super","semma","mass","love","awesome"]
neg_words = ["bad","waste","mokke","worst","hate"]

# -----------------------
# INLINE HIGHLIGHT
# -----------------------
def highlight_text(text):
    words = text.split()
    new = []
    for w in words:
        if w.lower() in pos_words:
            new.append(f"<span style='color:lightgreen;font-weight:bold'>{w}</span>")
        elif w.lower() in neg_words:
            new.append(f"<span style='color:#ff4d4d;font-weight:bold'>{w}</span>")
        else:
            new.append(w)
    return " ".join(new)

# -----------------------
# EXPLAIN WORDS
# -----------------------
def explain_words(text):
    words = text.lower().split()
    pos = [w for w in words if w in pos_words]
    neg = [w for w in words if w in neg_words]
    return pos, neg

# -----------------------
# PREDICT
# -----------------------
def predict(text):
    text_en = translate_text(text)

    v = vader.polarity_scores(text_en)
    b = bert(text_en)[0]

    pos_score = v['pos']
    neg_score = v['neg']
    neu_score = v['neu']
    comp = v['compound']

    sentiment = "Positive" if b['label']=="POSITIVE" else "Negative"

    if comp >= 0.6:
        sentiment = "Very Positive"
    elif comp <= -0.6:
        sentiment = "Very Negative"
    elif -0.05 < comp < 0.05:
        sentiment = "Neutral"

    if pos_score > 0 and neg_score > 0:
        sentiment = "Mixed"

    return sentiment, pos_score, neg_score, neu_score, comp, b

# -----------------------
# UI
# -----------------------
st.title("🔥 Smart Sentiment Analyzer")

text = st.text_area("Enter Tamil + English text")

if st.button("Analyze"):
    if text.strip()=="":
        st.warning("Enter text")
    else:
        with st.spinner("⚡ AI analyzing..."):
            sentiment, pos_s, neg_s, neu_s, comp, b = predict(text)

        st.subheader(f"Result: {sentiment}")

        # INLINE TEXT
        st.markdown("### 📝 Highlighted Text")
        st.markdown(highlight_text(text), unsafe_allow_html=True)

        # SCORES
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", round(pos_s,2))
        col2.metric("Negative", round(neg_s,2))
        col3.metric("Neutral", round(neu_s,2))

        st.metric("Compound Score", comp)
        st.write(f"BERT: {b['label']} ({b['score']:.2f})")

        # BAR
        fig, ax = plt.subplots()
        ax.bar(["Positive","Negative","Neutral"], [pos_s, neg_s, neu_s])
        st.pyplot(fig)

        # -----------------------
        # EXPLANATION UI
        # -----------------------
        st.markdown("### 🔍 Explanation")
        pos_w, neg_w = explain_words(text)

        if pos_w:
            st.markdown(f"""
            <div style='background:rgba(0,255,0,0.15); padding:15px; border-radius:10px'>
            <b>🟢 Positive Words:</b> {", ".join(pos_w)}<br>
            <span style='color:lightgreen'>These words caused positive sentiment</span>
            </div>
            """, unsafe_allow_html=True)

        if neg_w:
            st.markdown(f"""
            <div style='background:rgba(255,0,0,0.15); padding:15px; border-radius:10px'>
            <b>🔴 Negative Words:</b> {", ".join(neg_w)}<br>
            <span style='color:#ff4d4d'>These words caused negative sentiment</span>
            </div>
            """, unsafe_allow_html=True)

# -----------------------
# CSV
# -----------------------
st.write("## 📂 Upload CSV")

file = st.file_uploader("", type=["csv"])

if file:
    df = pd.read_csv(file)

    progress = st.progress(0)

    texts = df["text"].astype(str)

    scores = []
    sentiments = []

    for i, t in enumerate(texts):
        v = vader.polarity_scores(t)
        comp = v['compound']

        if comp >= 0.6:
            s = "Very Positive"
        elif comp <= -0.6:
            s = "Very Negative"
        elif -0.05 < comp < 0.05:
            s = "Neutral"
        else:
            s = "Positive" if comp > 0 else "Negative"

        scores.append(comp)
        sentiments.append(s)

        progress.progress((i+1)/len(texts))

    df["score"] = scores
    df["sentiment"] = sentiments

    st.success("✅ Fast Analysis Done")
    st.dataframe(df)

    counts = df["sentiment"].value_counts()

    # PIE
    st.write("### 🥧 Pie Chart")
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%')
    st.pyplot(fig1)

    # BAR
    st.write("### 📊 Bar Chart")
    fig2, ax2 = plt.subplots()
    counts.plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

    # HIST
    st.write("### 📈 Distribution")
    fig3, ax3 = plt.subplots()
    ax3.hist(df["score"], bins=10)
    st.pyplot(fig3)

    # TREND
    st.write("### 📉 Trend")
    df["index"] = range(len(df))
    fig4, ax4 = plt.subplots()
    ax4.plot(df["index"], df["score"])
    st.pyplot(fig4)