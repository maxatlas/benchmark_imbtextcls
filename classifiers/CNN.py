import torch
import math
import torch.nn as nn

from utils import get_label_ids


class TaskModel(nn.Module):
    def __init__(self, config):
        super(TaskModel, self).__init__()

        emb_weights = torch.load(config.emb_path)
        vocab_size, self.emb_d = emb_weights['weight'].shape

        self.emb = nn.Embedding(vocab_size, self.emb_d)
        self.emb.load_state_dict(emb_weights)

        self.num_labels = config.num_labels
        self.device = config.device
        self.num_layers = config.num_layers

    def freeze_emb(self):
        self.emb.weight.requires_grad = False

    def unfreeze_emb(self):
        self.emb.weight.requires_grad = True

    def _batch_eval(self, a, b, c, d, e):
        return

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            preds = self._batch_eval(input_ids, a, b, labels, label_names)
            if type(labels[0]) is list: preds = get_label_ids(preds, label_names)

        return preds


class Model(TaskModel):
    def __init__(self, config):
        super(CNN, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding = config.padding
        self.dilation = config.dilation
        self.word_max_length = config.n_positions
        self.stride = config.stride

        self.filters = config.filters
        # filters = [filter_size]
        self.n_filters = len(self.filters)
        self.conv_layers = [self._create_conv_layers(kernel_size) for kernel_size in self.filters]
        self.pool_layers = [self._create_pool_layer(kernel_size) for kernel_size in self.filters]

        self.cls_layer = nn.Linear(self._get_output_size(), self.num_labels)

    def _create_conv_layers(self, kernel_size):
        conv_layer = nn.Conv1d(self.emb_d, self.num_filters, kernel_size, self.stride)
        conv_layer.to(self.device)
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
        output_size = [_get_size(self.word_max_length, kernel_size)
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

    def batch_train(self, input_ids, label_ids, loss_func, a=None, b=None):
        logits, _ = self.forward(input_ids)

        loss = loss_func(logits, label_ids)
        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        _, preds = self.forward(input_ids)

        return preds
