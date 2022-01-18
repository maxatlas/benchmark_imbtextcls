import torch
import torch.nn as nn

from model_utils import pad_seq, batch_train, batch_eval
from transformers import (BertPreTrainedModel,
                          BertModel)


class Model(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.tokenizer = None

        if 'device' not in config.to_dict().keys():
            self.config.device = "cpu"

        self.bert = BertModel(config).to(self.config.device)
        classifier_dropout = config.hidden_dropout_prob \
            if not config.classifier_dropout else config.classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout).to(self.config.device)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(self.config.device)

        # Initialize weights and apply final processing
        self.init_weights()

    def freeze_emb(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

    def unfreeze_emb(self):
        for param in self.bert.embeddings.parameters():
            param.requires_grad = True

    def forward(self, texts, **kwargs):
        max_length = self.config.max_position_embeddings

        input_ids, token_type_ids, attention_mask = self.tokenizer.core(texts).values()

        input_ids = pad_seq(input_ids, max_length).to(self.config.device)
        attention_mask = pad_seq(attention_mask, max_length).to(self.config.device)
        token_type_ids = pad_seq(token_type_ids, max_length).to(self.config.device)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        preds = torch.argmax(logits, dim=1)
        
        return logits, preds

    def batch_train(self, texts, labels, label_names, loss_func, multi_label=False):
        self.bert = self.bert.to(self.config.device)
        self.dropout = self.dropout.to(self.config.device)
        self.classifier = self.classifier.to(self.config.device)
        return batch_train(self, texts, labels, label_names, loss_func, multi_label)

    def batch_eval(self, texts, labels, label_names, multi_label=False):
        return batch_eval(self, texts, labels, label_names, multi_label)
