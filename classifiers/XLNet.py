import torch
import torch.nn as nn

from utils import get_label_ids
from transformers import (XLNetPreTrainedModel,
                          XLNetModel,
                          modeling_utils)


class Model(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.transformer = XLNetModel(config)
        self.sequence_summary = modeling_utils.SequenceSummary(config)
        self.cls_layer = nn.Linear(config.d_model, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=None,
        **kwargs,  # delete when `use_cache` is removed in XLNetModel
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.cls_layer(output)
        preds = torch.argmax(logits, dim=1)

        return preds, logits

    def batch_train(self,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    label_ids,
                    loss_func,
                    **kwargs):
        _, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 **kwargs)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids,
                   attention_mask,
                   token_type_ids,
                   labels,
                   label_names,
                   **kwargs):
        with torch.no_grad():
            preds, _ = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    **kwargs)
            if type(labels[0]) is list: preds = get_label_ids(preds, label_names)
        return preds