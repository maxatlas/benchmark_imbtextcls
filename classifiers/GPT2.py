import torch
import torch.nn as nn

from model_utils import pad_seq, batch_train, batch_eval
from transformers import (GPT2PreTrainedModel,
                          GPT2Model)


class Model(GPT2PreTrainedModel):
    def __init__(self, config):

        super(Model, self).__init__(config)
        self.config = config
        self.tokenizer = None

        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config)
        dropout = config.resid_pdrop \
            if "dropout" not in config.to_dict() else config.dropout
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.n_embd, config.num_labels)

        self.cls_type = "last"

        self.init_weights()

    def set_cls_type(self, cls_type):
        self.cls_type = cls_type

    def forward(self, texts, **kwargs):
        max_length = self.config.n_positions

        input_ids, attention_mask = self.tokenizer.core(texts).values()

        input_ids = pad_seq(input_ids, max_length)
        attention_mask = pad_seq(attention_mask, max_length)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        if self.cls_type == "last":
            pooled_output = outputs[0][:, -1, :]
        elif self.cls_type == "first":
            pooled_output = outputs[0][:, 0, :]
        elif self.cls_type == "mean":
            pooled_output = torch.mean(outputs[0], dim=1)
        elif self.cls_type == "max":
            pooled_output = torch.max(outputs[0], dim=1)[0]
        elif self.cls_type == "min":
            pooled_output = torch.min(outputs[0], dim=1)[0]
        elif self.cls_type == "sum":
            pooled_output = torch.sum(outputs[0], dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, texts, labels, label_names, loss_func):
        return batch_train(self, texts, labels, label_names, loss_func)

    def batch_eval(self, texts, labels, label_names):
        return batch_eval(self, texts, labels, label_names)
