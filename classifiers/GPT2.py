import torch
import torch.nn as nn

from utils import get_label_ids, pad_seq_to_length
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
        self.score = nn.Linear(config.n_embd, config.num_labels)

        self.cls_type = "last"

        self.init_weights()

    def set_cls_type(self, cls_type):
        self.cls_type = cls_type

    def forward(self, texts, max_length=None):
        max_length = self.config.n_positions if not max_length else max_length
        input_ids, attention_mask = self.tokenizer.core(texts).values()
        input_ids = pad_seq_to_length(input_ids, max_length)
        attention_mask = pad_seq_to_length(attention_mask, max_length)
        input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)

        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
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
        logits = self.score(pooled_output)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, texts, labels, label_names, loss_func):
        label_ids = get_label_ids(labels, label_names)
        logits, _ = self.forward(texts)
        print(logits)
        print(label_ids)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids)
            if type(labels[0]) is list:
                preds = get_label_ids(preds, label_names)
        return preds


if __name__ == "__main__":
    from Config import ModelConfig, DataConfig
    from vars import datasets_meta
    from torch.utils.data import DataLoader
    import build_dataset

    mc = ModelConfig("gpt2", 1024, 2, 30)
    model = Model(mc())

    dc = DataConfig(*datasets_meta[-1].values())
    train_tds, _, _ = build_dataset.main(dc)
    train_dl = DataLoader(train_tds, batch_size=20)
    for b in train_dl:
        texts, labels = b
        model.forward(texts)