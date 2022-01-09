import torch
import torch.nn as nn

from model_utils import (batch_eval,
                         batch_train,
                         pad_seq)
from transformers import (RobertaPreTrainedModel,
                          RobertaModel)


class Model(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            texts,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        max_length = self.config.max_position_embeddings

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask = self.tokenizer.core(texts).values()
        input_ids = pad_seq(input_ids, max_length)
        attention_mask = pad_seq(attention_mask, max_length)

        input_ids, attention_mask = torch.tensor(input_ids),\
                                    torch.tensor(attention_mask)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        outputs = outputs[0][:, -1, :]  # take <s> token (equiv. to [CLS])
        outputs = self.dropout(outputs)

        logits = self.classifier(outputs)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, texts, labels, label_names, loss_func):
        return batch_train(self, texts, labels, label_names, loss_func)

    def batch_eval(self, texts, labels, label_names):
        return batch_eval(self, texts, labels, label_names)
