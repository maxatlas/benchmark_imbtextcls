import torch
import math
import torch.nn as nn

from model_utils import TaskModel, pad_seq_to_length


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.padding = config.padding
        self.dilation = config.dilation
        self.word_max_length = config.word_max_length
        self.stride = config.stride

        self.filters = torch.arange(2, config.num_layers + 2).tolist()

        self.n_filters = len(self.filters)
        self.conv_layers = nn.ModuleList([self._create_conv_layers(kernel_size) for kernel_size in self.filters])
        self.pool_layers = nn.ModuleList([self._create_pool_layer(kernel_size) for kernel_size in self.filters])

        self.cls = nn.Linear(self._get_output_size(), config.num_labels).to(config.device)

    def _create_conv_layers(self, kernel_size):
        conv_layer = nn.Conv1d(self.emb_d, self.n_filters, kernel_size, self.stride)
        conv_layer.to(self.config.device)
        return conv_layer

    def _create_pool_layer(self, kernel_size):
        pool_layer = nn.MaxPool1d(kernel_size, self.stride)
        return pool_layer

    def _get_output_size(self):
        def _get_size(length, kernel_size):

            out = math.floor((length + 2 * self.padding - self.dilation * (kernel_size - 1)
                              - 1) / self.stride + 1)
            return out

        output_size = [_get_size(self.word_max_length, kernel_size)
                       for kernel_size in self.filters]
        output_size = [_get_size(conv_size, kernel_size)
                       for kernel_size, conv_size in zip(self.filters, output_size)]
        output_size = sum(output_size) * len(self.filters)

        return output_size

    def forward(self, input_ids, **kwargs):
        input_ids = pad_seq_to_length(input_ids, self.config.word_max_length).to(self.config.device)
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

        logits = self.cls(out)
        preds = torch.argmax(logits, dim=1)
        return logits, preds
