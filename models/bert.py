from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from transformers import BertModel

class BERT_Model(nn.Module):
	def __init__(self, hidden_dim, p = 0.5):
		super().__init__()
		self.bert = BertModel.from_pretrained('bert-base-uncased') # pretrained bert model from hugging face
		self.dropout = nn.Dropout(p = 0.5)
		self.fc1 = nn.Linear(768, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim//4)
		self.fc3 = nn.Linear(hidden_dim//4, 1)

	def forward(self, ids, mask, token_type_ids):
		x = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)[1]
		x = self.fc1(x)
		x = self.dropout(10 * sigmoid(x))
		x = self.fc2(x)
		x = self.dropout(10 * sigmoid(x))
		x = self.fc3(x)
		x = 10 * sigmoid(x)
		return x