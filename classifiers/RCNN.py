import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import TaskModel, pad_seq


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.lstm = nn.LSTM(self.emb_d, config.cls_hidden_size,
                            bidirectional=True).to(self.config.device)
        self.W2 = nn.Linear(2 * config.cls_hidden_size + self.emb_d, config.cls_hidden_size).to(self.config.device)

    def forward(self, input_ids):
        input_ids = pad_seq(input_ids).to(self.config.device)

        x = self.emb(input_ids)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        output, (_, _) = self.lstm(x)

        final_encoding = torch.cat((output, x), 2).permute(1, 0, 2)
        for _ in range(self.config.num_layers):
            y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
            y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
            y = F.max_pool1d(y, y.size(2))  # y.size() = (batch_size, hidden_size, 1)
            y = y.squeeze(2)

        logits = self.cls(y)
        preds = torch.argmax(logits, dim=1)

        return logits, preds
