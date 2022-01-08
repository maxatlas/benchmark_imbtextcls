import torch
import torch.nn as nn

from utils import get_label_ids, pad_seq_to_length
from transformers import (BertPreTrainedModel,
                          BertModel)


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.tokenizer = None

        self.bert = BertModel(config)
        classifier_dropout = config.hidden_dropout_prob \
            if not config.classifier_dropout else config.classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(self, texts, max_length=None):
        max_length = self.config.n_positions if not max_length else max_length

        input_ids, attention_mask, token_type_ids = self.tokenizer(texts).values()
        input_ids = pad_seq_to_length(input_ids, max_length)
        attention_mask = pad_seq_to_length(attention_mask, max_length)
        input_ids, attention_mask = torch.tensor(input_ids), torch.tensor(attention_mask)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifierr(pooled_output)
        preds = torch.argmax(logits, dim=1)

        return preds, logits

    def batch_train(self,
                    texts,
                    label_ids,
                    loss_func):
        _, logits = self.forward(texts)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids,
                   attention_mask,
                   token_type_ids,
                   labels,
                   label_names):
        with torch.no_grad():
            preds, _ = self.forward(texts)
            if type(labels[0]) is list: preds = get_label_ids(preds, label_names)
        return preds
