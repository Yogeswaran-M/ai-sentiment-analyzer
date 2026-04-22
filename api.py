from fastapi import FastAPI
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

app = FastAPI()

# Models
vader = SentimentIntensityAnalyzer()
bert = pipeline("sentiment-analysis")

class TextRequest(BaseModel):
    text: str

# 🔥 Tamil + Emoji slang dictionary
custom_words = {
    "🔥": 2.5,
    "😍": 3.0,
    "😂": 1.5,
    "semma": 2.5,
    "vera level": 3.0,
    "mass": 2.5,
    "super": 2.0,
    "nalla": 2.0,
    "waste": -3.0,
    "mokke": -2.5,
    "kevalam": -3.0,
    "worst": -3.5,
    "sumar": -0.5
}
vader.lexicon.update(custom_words)

@app.post("/predict")
def predict(req: TextRequest):
    text = req.text

    # VADER score
    vader_score = vader.polarity_scores(text)['compound']

    # BERT result
    bert_result = bert(text)[0]
    bert_label = bert_result['label']
    bert_score = bert_result['score']

    # Combine logic
    if bert_label == "POSITIVE":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    if vader_score >= 0.6:
        sentiment = "Very Positive"
    elif vader_score <= -0.6:
        sentiment = "Very Negative"
    elif -0.05 < vader_score < 0.05:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "vader_score": vader_score,
        "bert_label": bert_label,
        "bert_confidence": bert_score
    }