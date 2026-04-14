import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    text = text.lower()
    return text