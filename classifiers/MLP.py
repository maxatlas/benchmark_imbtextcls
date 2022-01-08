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
        super(MLP, self).__init__(config)
        self.word_max_length = config.word_max_length
        self.hidden_size = config.hidden_size

        self.mlp0 = nn.Linear(self.emb_d, 1)
        self.mlp1 = nn.Linear(self.word_max_length, self.hidden_size)
        self.mlps = [nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
                     for _ in range(self.num_layers - 1)]
        self.cls = nn.Linear(self.hidden_size, self.num_labels)
        # self.layernorm = nn.LayerNorm()
        # TODO: Batchrnorm? layernorm? dropout?
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        embeds = self.emb(input_ids)
        out = self.mlp0(embeds).squeeze(2)
        out = self.mlp1(out)
        for mlp in self.mlps:
            out = mlp(out)
        logits = self.cls(out)
        preds = torch.argmax(logits, axis=1)

        return logits, preds

    def batch_train(self, input_ids, label_ids, loss_func, attention_mask=None, token_type_ids=None):
        logits, _ = self.forward(input_ids)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids)
        return preds