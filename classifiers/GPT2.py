import torch
import torch.nn as nn

from model_utils import pad_seq, batch_train, batch_eval
from classifiers.GPT2_homemade import (GPT2PreTrainedModel,
                                       GPT2Model)


class Model(GPT2PreTrainedModel):
    def __init__(self, config):

        super(Model, self).__init__(config)
        self.config = config
        self.tokenizer = None

        if 'device' not in config.to_dict().keys():
            self.config.device = "cpu"

        self.num_labels = config.num_labels

        self.transformer = GPT2Model(config).to(self.config.device)
        dropout = config.resid_pdrop \
            if "dropout" not in config.to_dict() else config.dropout
        self.dropout = nn.Dropout(dropout).to(self.config.device)
        self.classifier = nn.Linear(config.n_embd, config.num_labels).to(self.config.device)

        self.cls_type = "last"

    def freeze_emb(self):
        self.transformer.wte.weight.requires_grad = False
        self.transformer.wpe.weight.requires_grad = False

    def unfreeze_emb(self):
        self.transformer.wte.weight.requires_grad = True
        self.transformer.wpe.weight.requires_grad = True

    def set_cls_type(self, cls_type):
        self.cls_type = cls_type

    def forward(self, texts, **kwargs):
        max_length = self.config.n_positions

        out = self.tokenizer(texts)
        input_ids, attention_mask, token_type_ids = out['input_ids'], \
                                                    out.get("attention_mask"), \
                                                    out.get("token_type_ids")

        input_ids = pad_seq(input_ids, max_length).to(self.config.device)
        attention_mask = pad_seq(attention_mask, max_length).to(self.config.device)

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

    def batch_train(self, texts, labels, label_names, loss_func, multi_label=False):
        self.transformer = self.transformer.to(self.config.device)
        self.dropout = self.dropout.to(self.config.device)
        self.classifier = self.classifier.to(self.config.device)
        return batch_train(self, texts, labels, label_names, loss_func, multi_label)

    def batch_eval(self, texts, labels, label_names, multi_label=False):
        return batch_eval(self, texts, labels, label_names, multi_label)
