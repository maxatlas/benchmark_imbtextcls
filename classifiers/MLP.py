import torch
import torch.nn as nn

from model_utils import TaskModel, pad_seq_to_length


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.config = config

        self.mlp0 = nn.Linear(self.emb_d, 1)
        self.mlp1 = nn.Linear(config.word_max_length, config.cls_hidden_size)
        self.mlps = [nn.Linear(config.cls_hidden_size, config.cls_hidden_size)
                     for _ in range(config.num_layers - 1)]
        # self.layernorm = nn.LayerNorm()
        # TODO: Batch norm? layernorm? dropout?
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, device="cpu"):
        for layer in self.mlps:
            layer.to(device)
        input_ids = pad_seq_to_length(input_ids, self.config.word_max_length)

        embeds = self.emb(input_ids)
        out = self.mlp0(embeds).squeeze(2)
        out = self.mlp1(out)
        for mlp in self.mlps:
            out = mlp(out)
        logits = self.cls(out)
        preds = torch.argmax(logits, dim=1)

        return logits, preds
