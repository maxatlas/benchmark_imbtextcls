import torch
import torch.nn as nn

from model_utils import TaskModel, pad_seq


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.lstm = nn.LSTM(self.emb_d, config.cls_hidden_size, batch_first=True,
                            num_layers=config.num_layers,
                            dropout=config.dropout).to(config.device)
        self.dropout = nn.Dropout(config.dropout).to(config.device)

    def forward(self, input_ids):
        input_ids = pad_seq(input_ids, self.config.word_max_length).to(self.config.device)

        embeds = self.emb(input_ids)
        lstm_out, (ht, ct) = self.lstm(embeds)
        logits = self.cls(ht[-1])
        preds = torch.argmax(logits, dim=1)

        return logits, preds
