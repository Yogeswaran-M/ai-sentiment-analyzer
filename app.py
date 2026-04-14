import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ⚙️ Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# 🎨 PRO UI STYLE
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #2c5364, #1c1c1c);
    color: white;
}
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}
h1 {
    text-align: center;
    font-size: 42px;
    color: #00f2fe;
}
.stButton>button {
    background: linear-gradient(45deg, #00f2fe, #4facfe);
    color: white;
    border-radius: 25px;
    padding: 10px 25px;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px #00f2fe;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 15px;
    border: 1px dashed #00f2fe;
}
</style>
""", unsafe_allow_html=True)

# 🧠 TITLE
st.title("💬 Sentiment Analyzer")

# ✍️ TEXT INPUT
st.markdown('<div class="glass">', unsafe_allow_html=True)

text = st.text_area("Enter your text 👇")

def predict(text):
    text = text.lower()
    if "good" in text or "love" in text or "happy" in text:
        return "Positive"
    elif "bad" in text or "hate" in text or "sad" in text:
        return "Negative"
    else:
        return "Neutral"

if st.button("🔍 Analyze"):
    if text.strip() == "":
        st.warning("Enter something!")
    else:
        result = predict(text)
        
    if result.lower() == "positive":
        st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background-color:#00c853;color:white;text-align:center;font-size:18px;">
            {result}
        </div>
        """,
        unsafe_allow_html=True
    )

elif result.lower() == "negative":
    st.markdown(
        f"""
        <div style="padding:15px;border-radius:10px;background-color:#d50000;color:white;text-align:center;font-size:18px;">
            {result}
        </div>
        """,
        unsafe_allow_html=True
    )

else:  # Neutral
    st.markdown(
        f"""
        <div style="padding:15px;border-radius:10px;background-color:#ffd600;color:black;text-align:center;font-size:18px;">
            {result}
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown('</div>', unsafe_allow_html=True)

# 📂 CSV UPLOAD
st.markdown('<div class="glass">', unsafe_allow_html=True)

st.markdown("### 📂 Upload CSV for Bulk Analysis")

file = st.file_uploader("", type=["csv"])

if file is not None:

    # 🔥 Accurate file size
    file_bytes = len(file.getvalue())
    size_text = f"{round(file_bytes/1024,2)} KB"

    st.success(f"✅ {file.name} ({size_text})")

    try:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")

        else:
            # ⚡ ULTRA FAST VECTORIZED METHOD
            df["text"] = df["text"].astype(str).str.lower()

            df["sentiment"] = "Neutral"

            df.loc[df["text"].str.contains("good|love|happy"), "sentiment"] = "Positive"
            df.loc[df["text"].str.contains("bad|hate|sad"), "sentiment"] = "Negative"

            st.success("⚡ Analysis completed instantly!")

            st.write("### 📊 Analyzed Data")
            st.dataframe(df)

            # 📊 Pie chart
            st.write("### 📈 Sentiment Distribution")

            counts = df["sentiment"].value_counts()

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown('</div>', unsafe_allow_html=True)