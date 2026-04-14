import os
from dotenv import load_dotenv

load_dotenv()

import requests

API_URL = "https://router.huggingface.co/hf-inference/models/cardiffnlp/twitter-roberta-base-sentiment"

API_TOKEN =os.getenv("API_TOKEN")  # 👈 உன் token

headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

def predict(text):
    response = requests.post(API_URL, headers=headers, json={"inputs": text})

    # 👇 safe check
    if response.status_code != 200:
        return f"Error: {response.text}"

    try:
        result = response.json()
    except:
        return "Error: Invalid response from API"

    print(result)

    try:
        label = result[0][0]['label']

        if label == "LABEL_0":
            return "Negative"
        elif label == "LABEL_1":
            return "Neutral"
        elif label == "LABEL_2":
            return "Positive"
        else:
            return label

    except:
        return f"Error: {result}"