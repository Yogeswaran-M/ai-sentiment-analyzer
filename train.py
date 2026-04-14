import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from model.model import SentimentModel

class Data(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['text']
        self.labels = df['label']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return enc['input_ids'].squeeze(), enc['attention_mask'].squeeze(), torch.tensor(self.labels[i])

df = pd.read_csv("data/dataset.csv")

label_map = {"negative":0,"neutral":1,"positive":2}
df['label'] = df['label'].map(label_map)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

loader = DataLoader(Data(df, tokenizer), batch_size=8)

model = SentimentModel()
opt = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    for ids, mask, label in loader:
        out = model(ids, mask)
        loss = loss_fn(out, label)

        opt.zero_grad()
        loss.backward()
        opt.step()

torch.save(model.state_dict(), "model.pth")