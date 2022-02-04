import torch
import torch.nn as nn

from model_utils import TaskModel, pad_seq


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.config = config

        self.mlp0 = nn.Linear(self.emb_d, config.cls_hidden_size).to(self.config.device)
        self.mlps = nn.ModuleList(nn.Linear(config.cls_hidden_size, config.cls_hidden_size).to(self.config.device)
                                  for _ in range(config.num_layers - 1))
        # self.layernorm = nn.LayerNorm()
        # TODO: Batch norm? layernorm? dropout?
        self.dropout = nn.Dropout(config.dropout).to(self.config.device)

    def forward(self, input_ids):
        input_ids = pad_seq(input_ids).to(self.config.device)

        embeds = self.emb(input_ids)
        out = self.mlp0(embeds)

        for mlp in self.mlps:
            out = mlp(out)
        logits = self.cls(out)
        preds = torch.argmax(logits, dim=1)

        return logits, preds
    