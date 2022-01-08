import torch
import torch.nn as nn

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
        super(LSTM, self).__init__(config)
        self.lstm = nn.LSTM(self.emb_d, config.hidden_size, batch_first=True,
                            num_layers=config.n_layers,
                            dropout=config.dropout)
        self.cls = nn.Linear(config.hidden_size, config.n_labels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        embeds = self.emb(input_ids)
        lstm_out, (ht, ct) = self.lstm(embeds)
        logits = self.cls(ht[-1])
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, input_ids, a, b, label_ids, loss_func):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def _batch_eval(self, input_ids, a,b, labels, label_names):
        _, preds = self.forward(input_ids)
        return preds