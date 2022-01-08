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
        super(LSTMattn, self).__init__(config)
        self.hidden_size = config.hidden_size

        self.lstm = nn.LSTM(self.emb_d, self.hidden_size, batch_first=True, num_layers=config.num_layers, )
        self.cls = nn.Linear(self.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def pay_attn(self, lstm_output):
        f, (hx, _) = lstm_output
        attn_weights = torch.bmm(f, hx.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, 1)
        weighted_f = torch.bmm(f.transpose(1, 2), soft_attn_weights).squeeze(2)

        return weighted_f

    def forward(self, input_ids):
        embeds = self.emb(input_ids)
        out = self.lstm(embeds)
        out = self.pay_attn(out)

        logits = self.cls(out)
        preds = torch.argmax(logits, dim=1) # TODO dim index out of range??

        return logits, preds

    def batch_train(self, input_ids, a, b, label_ids, loss_func):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def _batch_eval(self, input_ids, a, b, labels, label_names):
        _, preds = self.forward(input_ids)

        return preds
