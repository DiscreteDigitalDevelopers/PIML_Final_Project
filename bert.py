#!/usr/bin/env python
# coding: utf-8

# # Bert Model

# In[1]:


import numpy as np
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as du
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel, BertConfig


# ## Dataset Class

# In[2]:


class Movie_Review_Data(Dataset):
    '''
    data_path: location of dataset
    seq_len: maximum length of a sentence
    embeddings_size: length of a word embedding vector
    '''
    def __init__(self, data_path, seq_len, tokenizer, num_splits):
        super(Movie_Review_Data, self).__init__()
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        data_dict = None
        with open(data_path, 'rb') as handle:
            data_dict = pickle.load(handle)
        if(data_dict is None):
            return "Invalid data path"
        self.data = []
        self.labels = []
        for d in data_dict.items():
            words = d[1:][0][1].split()
            label = int(np.round(d[1:][0][0])-1)
            length = max(len(words), seq_len)
            for i in range(num_splits):
                if(i*(length//num_splits)+seq_len > length):
                    next_words = words[i*(length//num_splits):]
                else:
                    next_words = words[i*(length//num_splits):i*(length//num_splits)+seq_len]
                if(len(next_words) > 0):
                    self.data.append(next_words)
                    # self.labels.append(1 if label > 6 else 0)
                    self.labels.append(label)

    def __len__(self):
        '''return len of dataset'''
        return len(self.data)
        
    def __getitem__(self, idx):
        '''return sequence, future sequence'''
        text = str(self.data[idx])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.seq_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ## Model Creation

# In[3]:


class BERT(torch.nn.Module):
    def __init__(self, dropout, out_dim):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') # pretrained bert model from hugging face
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(768, 768//2)
        self.fc2 = torch.nn.Linear(768//2, out_dim)
    
    def forward(self, ids, mask, token_type_ids):
        output = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output = self.dropout1(output[1])
        output = self.fc1(output)
        output = self.fc2(self.dropout2(F.relu(output)))
        return output
        
    def checkpoint(self, checkpoint_pth, epoch, train_loss_list, valid_loss_list, optimizer):
        checkpoint = {
            'epoch': epoch,
            'train_loss': train_loss_list,
            'valid_loss': valid_loss_list,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_pth)


# # Hyperparameters and Instantiating Model

# In[4]:


device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")
batch_size = 4
learning_rate = 1e-5
max_epochs = 100
dropout = 0.3
out_dim = 10
seq_len = 512
num_splits = 5
seed = 0
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

torch.manual_seed(seed)
model = BERT(dropout, out_dim)
# checkpoint = torch.load("bert.pth")
# model.load_state_dict(checkpoint["state_dict"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# load training data in batches
SAVE_LOCATION = './data/'
train_loader = du.DataLoader(dataset=Movie_Review_Data(f'{SAVE_LOCATION}processed/data_train.pkl', seq_len, tokenizer, num_splits),
                             batch_size=batch_size,
                             shuffle=True)
dev_loader = du.DataLoader(dataset=Movie_Review_Data(f'{SAVE_LOCATION}processed/data_dev.pkl', seq_len, tokenizer, num_splits),
                             batch_size=batch_size,
                             shuffle=True)
test_loader = du.DataLoader(dataset=Movie_Review_Data(f'{SAVE_LOCATION}processed/data_test.pkl', seq_len, tokenizer, num_splits),
                             batch_size=batch_size,
                             shuffle=True)
# send model over to device
model = model.to(device)
model.train()


# ## Training Model

# In[5]:



last_loss = np.inf
train_loss_list = []
valid_loss_list = []
# iterating through all epochs
for epoch in range(1, max_epochs + 1):    
    # training step
    train_loss = 0.
    train_accuracy = 0.
    # model.train()
    # iterating through entire dataset in batches
    for batch_idx, data in enumerate(tqdm(train_loader)):
        # sending batch over to device
        ids, mask, token_type_ids, targets = data["ids"].to(device), data["mask"].to(device), data["token_type_ids"].to(device), data["targets"].to(device)
        optimizer.zero_grad()
        # getting predictions from model
        pred = model(ids, mask, token_type_ids)
        # # calculating BCE loss between predictions and labels
        loss = F.cross_entropy(pred, targets)
        train_loss += loss.item()
        # # calculating backprop and using an adam optimizer for update step 
        loss.backward()
        optimizer.step()
        train_accuracy += torch.sum(torch.argmax(pred, dim=1) == targets)
    dev_loss = 0.
    dev_accuracy = 0.
    pred_list = []
    target_list = []
    with torch.no_grad():
        model.eval()
        # iterating through entire dataset in batches
        for batch_idx, data in enumerate(tqdm(dev_loader)):
            # sending batch over to device
            ids, mask, token_type_ids, targets = data["ids"].to(device), data["mask"].to(device), data["token_type_ids"].to(device), data["targets"].to(device)
            # zeroing out previous gradients
            optimizer.zero_grad()
            # getting predictions from model
            pred = model(ids, mask, token_type_ids)
            # calculating BCE loss between predictions and labels
            loss = F.cross_entropy(pred, targets)
            dev_loss += loss.item()
            dev_accuracy += torch.sum(torch.argmax(pred, dim=1) == targets)
            pred_list.append(torch.argmax(pred, dim=1))
            target_list.append(targets)
    train_loss /= len(train_loader.dataset)
    train_accuracy /= len(train_loader.dataset)
    dev_loss /= len(dev_loader.dataset)
    dev_accuracy /= len(dev_loader.dataset)
    train_loss_list.append(train_loss)
    valid_loss_list.append(dev_loss)
    model.checkpoint("bert.pth", epoch, train_loss_list, valid_loss_list, optimizer)
    print(f"Epoch: {epoch}, training_loss {train_loss}, training_accuracy {train_accuracy}, dev_loss {dev_loss}, dev_accuracy {dev_accuracy}")
        


# In[ ]:




