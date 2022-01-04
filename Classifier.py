import math
import torch
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
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

    def batch_eval(self, input_ids, labels, label_names):
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

        print(output)

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

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 2 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embedding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 

        """

        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.lstm = nn.LSTM(self.emb_d, self.hidden_size, dropout=config.hidden_dropout_prob, bidirectional=True)
        self.W2 = nn.Linear(2 * self.hidden_size + self.emb_d, self.hidden_size)
        self.cls_layer = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, batch_size=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        """

        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
        of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
        its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
        state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
        vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
        dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.
        """
        x = self.emb(
            input_ids)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        x = x.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        output, (final_hidden_state, final_cell_state) = self.lstm(x)

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
