import torch
import torch.nn as nn

from model_utils import (pad_seq,
                         batch_eval,
                         batch_train)
from classifiers.XLNet_homemade import (XLNetPreTrainedModel,
                                        XLNetModel)
from transformers import modeling_utils


class Model(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if 'device' not in config.to_dict().keys():
            self.config.device = "cpu"

        self.tokenizer = None

        self.transformer = XLNetModel(config).to(self.config.device)
        self.sequence_summary = modeling_utils.SequenceSummary(config).to(self.config.device)
        self.classifier = nn.Linear(config.d_model, config.num_labels).to(self.config.device)

    def freeze_emb(self):
        self.transformer.word_embedding.weight.requires_grad = False

    def unfreeze_emb(self):
        self.transformer.word_embedding.weight.requires_grad = True

    def forward(
        self,
        texts,
        return_dict=None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        out = self.tokenizer.core(texts)
        input_ids, attention_mask, token_type_ids = out['input_ids'], \
                                                    out.get("attention_mask"), \
                                                    out.get("token_type_ids")

        input_ids = pad_seq(input_ids).to(self.config.device)
        attention_mask = pad_seq(attention_mask).to(self.config.device)
        if token_type_ids:
            token_type_ids = pad_seq(token_type_ids).to(self.config.device)

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.classifier(output)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, texts, labels, label_names, loss_func, multi_label=False):
        self.transformer = self.transformer.to(self.config.device)
        self.sequence_summary = self.sequence_summary.to(self.config.device)
        self.classifier = self.classifier.to(self.config.device)
        return batch_train(self, texts, labels, label_names, loss_func, multi_label)

    def batch_eval(self, texts, labels, label_names, multi_label=False):
        return batch_eval(self, texts, labels, label_names, multi_label)
