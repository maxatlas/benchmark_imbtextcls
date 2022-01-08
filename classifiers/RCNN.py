import torch
import torch.nn as nn

import torch.nn.functional as F
from utils import get_label_ids


class TaskModel(nn.Module):
    def __init__(self, config):
        super(TaskModel, self).__init__()

        emb_weights = torch.load(config.emb_path)
        vocab_size, self.emb_d = emb_weights['weight'].shape

        self.emb = nn.Embedding(vocab_size, self.emb_d)
        self.emb.load_state_dict(emb_weights)

        self.num_labels = config.num_labels
        self.device = config.device
        self.num_layers = config.num_layers

    def freeze_emb(self):
        self.emb.weight.requires_grad = False

    def unfreeze_emb(self):
        self.emb.weight.requires_grad = True

    def _batch_eval(self, a, b, c, d, e):
        return

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            preds = self._batch_eval(input_ids, a, b, labels, label_names)
            if type(labels[0]) is list: preds = get_label_ids(preds, label_names)

        return preds


class Model(TaskModel):
    def __init__(self, config):
        super(RCNN, self).__init__(config)
        self.hidden_size = config.hidden_size

        self.lstm = nn.LSTM(self.emb_d, self.hidden_size, num_layers=config.num_layers,
                            dropout=config.hidden_dropout_prob, bidirectional=True)
        self.W2 = nn.Linear(2 * self.hidden_size + self.emb_d, self.hidden_size)
        self.cls_layer = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        output, (_, _) = self.lstm(x)

        final_encoding = torch.cat((output, x), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size(2))  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        logits = self.cls_layer(y)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, input_ids, a, b, label_ids, loss_func):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def batch_eval(self, input_ids, labels, label_names, a=None, b=None):
        _, preds = self.forward(input_ids)

        return preds

