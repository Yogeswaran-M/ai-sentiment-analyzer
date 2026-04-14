import torch
import torch.nn as nn
from transformers import BertModel

class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.cnn = nn.Conv1d(768, 256, 3)
        self.rnn = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 3)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0,2,1)
        x,_ = self.rnn(x)
        x = x[:,-1,:]
        return self.fc(x)