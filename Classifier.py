import math
import torch
import torch.nn as nn

import numpy as np

from transformers import (GPT2PreTrainedModel, GPT2Model,
                          BertPreTrainedModel, BertModel,
                          XLNetPreTrainedModel, XLNetModel)

from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from gensim.models.keyedvectors import KeyedVectors

torch.manual_seed(0)
"""
    Deep Learning classifier:
        DNN
        RNN (GRU/LSTM)
        CNN
        Hierarchical Attention Networks
        RCNN
        RMDL
        HDLTex
"""


def transform_labels(labels, label_no):
    labels_ids = []
    for label in labels:
        ids = [0] * label_no
        ids[label] = 1
        labels_ids.append(ids)
    return labels_ids


def build_emb_layer(tknwords:set, kv: KeyedVectors, trainable=1):
    def _create_weight_matrix():
        wm = np.zeros((num_emb, emb_dim))
        for i, word in enumerate(tknwords[1:]):
            try:
                wm[i+1] = kv[word]
            except KeyError:
                wm[i+1] = np.random.normal(scale=0.6, size=(emb_dim, ))
                unfound_words.add(word)
        wm = torch.tensor(wm)
        return wm

    unfound_words = set()
    num_emb, emb_dim = len(tknwords), len(kv['the'])
    emb_layer = nn.Embedding(num_emb, emb_dim)
    emb_layer.load_state_dict({'weight': _create_weight_matrix()})
    if not trainable: emb_layer.weight.requires_grad = False

    return emb_layer, unfound_words


def metrics_frame(preds, labels, label_names):
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    cr = classification_report(labels, preds,)
                               # labels=list(range(len(label_names))), target_names=label_names)
    model_metrics = {"Precision, Micro": precision_micro, "Precision, Macro": precision_macro,
                     "Recall, Micro": recall_micro, "Recall, Macro": recall_macro,
                     "F1 score, Micro": f1_micro, "F1 score, Macro": f1_macro, "Classification report": cr}
    return model_metrics


class GPT2(GPT2PreTrainedModel):
    def __init__(self, label_no, config):

        super(GPT2, self).__init__(config)

        self.cls_no = label_no
        self.n_positions = config.n_positions

        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(config.hidden_size, label_no)

        self.cls_type = "max"

        self.init_weights()

    def set_cls_type(self, cls_type):
        self.cls_type = cls_type

    def forward(self, input_ids):
        outputs = self.gpt2(
            input_ids
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
        logits = self.cls_layer(pooled_output)
        preds = torch.argmax(logits, axis=1)

        return logits, preds

    def batch_train(self, input_ids, attention_mask, token_type_ids, label_ids, loss_func):
        logits, _ = self.forward(input_ids)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids)
        return metrics_frame(preds, labels, label_names)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def batch_train(self,
                    loss_funct,
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,):
        logits = self.forward()
        loss = loss_funct(logits, labels)

        return loss


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, sentence):
        embeds = sentence
        lstm_out, (ht, ct) = self.lstm(embeds.view(len(sentence), 1, -1))
        linear_out = self.hidden2label(ht[-1])

        return linear_out


class CNNClassifier(nn.Module):
    def __init__(self, params):
        super(CNNClassifier, self).__init__()
        self.seq_len = params.seq_len
        self.num_words = params.num_words
        self.embedding_size = params.embedding_size

        self.dropout = nn.Dropout(0.25)

        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # number of output channels of the convolution for each layer.
        self.out_size = params.out_size
        self.stride = params.stride

        self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0) # vocab size + 1 for padding

        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)

        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.fc = nn.Linear(self.in_features_fc(), 1)

    def in_features_fc(self):
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        out_conv_2 = (())
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)


