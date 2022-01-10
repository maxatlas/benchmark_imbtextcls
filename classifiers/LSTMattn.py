import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import TaskModel, pad_seq


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)

        self.lstm = nn.LSTM(self.emb_d, config.cls_hidden_size,
                            batch_first=True, num_layers=config.num_layers, ).to(self.config.device)
        self.cls = nn.Linear(config.cls_hidden_size, config.num_labels).to(self.config.device)
        self.dropout = nn.Dropout(config.dropout).to(self.config.device)

    @staticmethod
    def pay_attn(lstm_output):
        f, (hx, _) = lstm_output
        attn_weights = torch.bmm(f, hx.permute(1, 2, 0))
        soft_attn_weights = F.softmax(attn_weights, 1)
        weighted_f = torch.bmm(f.transpose(1, 2), soft_attn_weights).squeeze(2)

        return weighted_f

    def forward(self, input_ids):
        input_ids = pad_seq(input_ids, self.config.word_max_length).to(self.config.device)

        embeds = self.emb(input_ids)
        out = self.lstm(embeds)
        out = self.pay_attn(out)

        logits = self.cls(out)
        preds = torch.argmax(logits, dim=1)  # TODO dim index out of range??

        return logits, preds
