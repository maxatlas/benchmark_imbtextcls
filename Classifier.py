import math
import torch
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
from utils import matrix_mul, element_wise_mul
from torch.nn import functional as F
from transformers import (GPT2PreTrainedModel, GPT2Model,
                          BertPreTrainedModel, BertModel,
                          XLNetPreTrainedModel, XLNetModel,
                          RobertaPreTrainedModel, RobertaModel,
                          modeling_utils)

from sklearn.metrics import (f1_score,
                             roc_curve,
                             auc,
                             recall_score,
                             precision_score,
                             classification_report)
from gensim.models.keyedvectors import KeyedVectors

"""
    Deep Learning classifier:
        DNN
        RNN (GRU/LSTM)
        CNN
        Hierarchical Attention Networks
        RCNN
        RMDL
        HDLTex
        Roberta
        BERT
        GPT
        XLNet
"""


def transform_labels(labels, label_no):
    labels_ids = []
    for label in labels:
        ids = [0] * label_no
        ids[label] = 1
        labels_ids.append(ids)
    return labels_ids


def build_emb_layer(tknwords:set, kv: KeyedVectors, trainable=1):
    def _create_weight_matrix(start_i):
        wm = np.zeros((num_emb, emb_dim))
        for i, word in enumerate(tknwords[start_i:]):
            try:
                wm[i+start_i] = kv[word]
            except KeyError:
                wm[i+start_i] = np.random.normal(scale=0.6, size=(emb_dim, ))
                unfound_words.add(word)
        wm = torch.tensor(wm)
        return wm

    unfound_words = set()
    num_emb, emb_dim = len(tknwords), len(kv['the'])
    word_start_i = 2
    emb_layer = nn.Embedding(num_emb, emb_dim)
    emb_layer.load_state_dict({'weight': _create_weight_matrix(word_start_i)})
    if not trainable: emb_layer.weight.requires_grad = False

    return emb_layer, unfound_words


def metrics_frame(preds, labels, label_names):
    recall_micro = recall_score(labels, preds, average="micro")
    recall_macro = recall_score(labels, preds, average="macro")
    precision_micro = precision_score(labels, preds, average="micro")
    precision_macro = precision_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    auc_res = auc(fpr, tpr)
    cr = classification_report(labels, preds,)
                               # labels=list(range(len(label_names))), target_names=label_names)

    model_metrics = {
        "Precision, Micro": precision_micro,
        "Precision, Macro": precision_macro,
        "Recall, Micro": recall_micro,
        "Recall, Macro": recall_macro,
        "F1 score, Micro": f1_micro,
        "F1 score, Macro": f1_macro,
        "ROC curve": (fpr, tpr, thresholds),
        "AUC": auc_res,
        "Classification report": cr,
    }

    return model_metrics


class TaskModel(nn.Module):
    def __init__(self, config):
        super(TaskModel, self).__init__()

        emb_weights = torch.load(config.emb_path)
        vocab_size, self.emb_d = emb_weights['weight'].shape

        self.emb = nn.Embedding(vocab_size, self.emb_d)
        self.emb.load_state_dict(emb_weights)

    def freeze_emb(self):
        self.emb.weight.requires_grad = False

    def unfreeze_emb(self):
        self.emb.weight.requires_grad = True


class GPT2(GPT2PreTrainedModel):
    def __init__(self, config):

        super(GPT2, self).__init__(config)

        self.cls_no = config.num_labels
        self.n_positions = config.n_positions

        self.model = GPT2Model(config)
        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(config.hidden_size, self.cls_no)

        self.cls_type = "max"

        self.init_weights()

    def set_cls_type(self, cls_type):
        self.cls_type = cls_type

    def forward(self, input_ids):
        outputs = self.model(
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

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids)
        return metrics_frame(preds, labels, label_names)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Roberta(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.init_weights()

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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        logits = self.classifier(outputs[0])
        preds = torch.argmax(logits, axis=1)

        return logits, preds

    def batch_train(self, input_ids, attention_mask, token_type_ids, label_ids, loss_func):
        logits, _ = self.forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids, attention_mask, token_type_ids, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids, attention_mask, token_type_ids)
        return metrics_frame(preds, labels, label_names)


class BERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.model = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls_layer = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.cls_layer(pooled_output)
        preds = torch.argmax(logits, axis=1)

        return preds, logits

    def batch_train(self,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    label_ids,
                    loss_func):
        _, logits = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids,
                   attention_mask,
                   token_type_ids,
                   labels,
                   label_names):
        preds, _ = self.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        res = metrics_frame(preds, labels, label_names)
        return res


class XLNet(XLNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.cls_no = config.num_labels
        self.config = config

        self.model = XLNetModel(config)
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

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs,
        )
        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.cls_layer(output)
        preds = torch.argmax(logits, axis=1)

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
        preds, _ = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                **kwargs)
        res = metrics_frame(preds, labels, label_names)
        return res


class LSTM(TaskModel):
    def __init__(self, config):
        super(LSTM, self).__init__(config)
        self.lstm = nn.LSTM(self.emb_d, config.hidden_size, batch_first=True, dropout=config.hidden_dropout_prob)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        embeds = self.emb(input_ids)
        lstm_out, (ht, ct) = self.lstm(embeds)
        logits = self.cls(ht[-1])
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, input_ids, a, b, label_ids, loss_func):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def batch_eval(self, input_ids, a,b, labels, label_names):
        _, preds = self.forward(input_ids)

        res = metrics_frame(preds, labels, label_names)
        return res


class RCNN(TaskModel):
    def __init__(self, config):
        super(RCNN, self).__init__(config)

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.lstm = nn.LSTM(self.emb_d, self.hidden_size,
                            dropout=config.hidden_dropout_prob, bidirectional=True)
        self.W2 = nn.Linear(2 * self.hidden_size + self.emb_d, self.hidden_size)
        self.cls_layer = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids):
        x = self.emb(input_ids)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        output, (_, _) = self.lstm(x)

        final_encoding = torch.cat((output, x), 2).permute(1, 0, 2)
        y = self.W2(final_encoding)  # y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0, 2, 1)  # y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y, y.size(2))  # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)

        logits = self.cls_layer(y)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, input_ids, a, b, label_ids, loss_func):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        _, preds = self.forward(input_ids)

        res = metrics_frame(preds, labels, label_names)
        return res


class CNN(TaskModel):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_filters = config.num_filters
        self.padding = config.padding
        self.dilation = config.dilation
        self.max_length = config.n_positions
        self.num_labels = config.num_labels
        self.stride = config.stride

        self.filters = config.filters
        # filters = [filter_size]
        self.conv_layers = [self._create_conv_layers(kernel_size) for kernel_size in self.filters]
        self.pool_layers = [self._create_pool_layer(kernel_size) for kernel_size in self.filters]

        self.cls_layer = nn.Linear(self._get_output_size(), self.num_labels)

    def _create_conv_layers(self, kernel_size):
        conv_layer = nn.Conv1d(self.emb_d, self.num_filters, kernel_size, self.stride)
        return conv_layer

    def _create_pool_layer(self, kernel_size):
        pool_layer = nn.MaxPool1d(kernel_size, self.stride)
        return pool_layer

    def _get_output_size(self):
        def _get_size(length, kernel_size):
            out = math.floor((length + 2 *
                        self.padding - self.dilation * (kernel_size - 1)
                        - 1) / self.stride + 1)
            return out
        output_size = [_get_size(self.max_length, kernel_size)
                       for kernel_size in self.filters]
        output_size = [_get_size(conv_size, kernel_size)
                       for kernel_size, conv_size in zip(self.filters, output_size)]
        output_size = sum(output_size) * len(self.filters)

        return output_size

    def forward(self, input_ids):
        embeds = self.emb(input_ids)
        # (N, L, D) -> (N, D, L)
        embeds = embeds.permute(0, 2, 1)
        outs = []
        for conv_layer, pool_layer in zip(self.conv_layers, self.pool_layers):
            out = conv_layer(embeds)
            out = torch.relu(out)
            out = pool_layer(out)
            outs.append(out)

        out = torch.cat(outs, dim=2)
        out = out.reshape(out.size(0), -1)

        logits = self.cls_layer(out)
        preds = torch.argmax(logits, dim=1)

        return logits, preds

    def batch_train(self, input_ids, a, b, label_ids, loss_func):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        _, preds = self.forward(input_ids)

        res = metrics_frame(preds, labels, label_names)
        return res


class HAN(TaskModel):
    def __init__(self, config):
        super(HAN, self).__init__(config)
        self.batch_size = config.batch_size
        self.num_labels = config.num_labels
        self.word_hidden_size = config.word_hidden_size
        self.sent_hidden_size = config.sent_hidden_size

        self.word_hidden_state, self.sent_hidden_state = None, None

        self.word_weight = nn.Parameter(torch.Tensor(2 * self.word_hidden_size, 2 * self.word_hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * self.word_hidden_size))
        self.word_context_weight = nn.Parameter(torch.Tensor(2 * self.word_hidden_size, 1))

        self.sent_weight = nn.Parameter(torch.Tensor(2 * config.sent_hidden_size, 2 * config.sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * config.sent_hidden_size))
        self.sent_context_weight = nn.Parameter(torch.Tensor(2 * config.sent_hidden_size, 1))

        self.word_gru = nn.GRU(self.emb_d, self.word_hidden_size,
                               bidirectional=True, dropout=config.hidden_dropout_prob)
        self.sent_gru = nn.GRU(2 * self.word_hidden_size, self.sent_hidden_size,
                               bidirectional=True, dropout=config.hidden_dropout_prob)
        self.cls = nn.Linear(2 * self.sent_hidden_size, self.num_labels)

        self._create_weights()
        self._init_hidden_state()

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.word_context_weight.data.normal_(mean, std)
        self.sent_weight.data.normal_(mean, std)
        self.sent_context_weight.data.normal_(mean, std)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        # if torch.cuda.is_available():
        #     self.word_hidden_state = self.word_hidden_state.cuda()
        #     self.sent_hidden_state = self.sent_hidden_state.cuda()

    def forward(self, input_ids):
        sent_list = []

        input_ids = input_ids.permute(1, 0, 2)
        for sent in input_ids:
            sent = sent.permute(1, 0)
            embeds = self.emb(sent)
            f_output, h_output = self.word_gru(embeds.float(), self.word_hidden_state)
            output = matrix_mul(f_output, self.word_weight, self.word_bias)
            output = matrix_mul(output, self.word_context_weight).permute(1, 0)
            output = F.softmax(output, dim=1)
            output = element_wise_mul(f_output, output.permute(1, 0))

            self.word_hidden_state = h_output

            sent_list.append(output)

        output = torch.cat(sent_list, 0)

        f_output, h_output = self.sent_gru(output, self.sent_hidden_state)
        self.sent_hidden_state = h_output

        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.sent_context_weight).permute(1, 0)
        output = F.softmax(output, dim=1)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        logits = self.cls(output)
        preds = torch.argmax(logits, axis=1)

        return logits, preds

    def batch_train(self, input_ids, attention_mask, token_type_ids, label_ids, loss_func):
        logits, _ = self.forward(input_ids)
        print(logits, label_ids)
        print(logits.shape, label_ids.shape)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids)
        return metrics_frame(preds, labels, label_names)

