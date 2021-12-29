from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from Classifier import GPT2, transform_labels
from TaskDataset import TaskDataset, split_tds
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from tqdm import tqdm

max_length=1024
# tds = TaskDataset().load("dataset/imdb.tds")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
train_set, test_set, val_set = split_tds("dataset/imdb.tds")

label_ids = torch.tensor(transform_labels(train_set.labels, len(train_set.labels_meta.names)), dtype=float)
input_ids = tokenizer(train_set.data)['input_ids']
input_ids = [torch.tensor(i[:max_length]) for i in input_ids]
input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

train_set = TensorDataset(input_ids, label_ids)
train_set = DataLoader(train_set, batch_size=3, shuffle=True)

model_config = GPT2Config()
model = GPT2(2, 768, model_config)

model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

loss_func = CrossEntropyLoss()
tr_loss = 0

for step, batch in enumerate(tqdm(train_set, desc="Iteration")):
    input_ids, label_ids = batch
    loss = model.batch_train(input_ids, label_ids, loss_func)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    tr_loss += loss.item()
    print(tr_loss)

import numpy as np
n=3
from Classifier import metrics_frame
logits, preds = model.forward(input_ids[:n])
preds = preds.detach().cpu().numpy()
labels = np.array(test_set.labels[:n])
label_names = test_set.labels_meta.names
out = metrics_frame(preds, labels, label_names)
# model.eval(input_ids[:n], label_ids[:n], test_set.labels_meta.names)