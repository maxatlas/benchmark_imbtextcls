import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import TaskModel, pad_seq_han


def matrix_mul(input, weight, bias=False):

    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0)


def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 1).unsqueeze(0)


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.num_labels = config.num_labels
        self.word_hidden_size = config.word_hidden_size
        self.sent_hidden_size = config.sent_hidden_size

        self.word_hidden_state, self.sent_hidden_state = None, None

        self.word_weight = nn.Parameter(torch.Tensor(2 * self.word_hidden_size,
                                                     2 * self.word_hidden_size)).to(self.config.device)
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * self.word_hidden_size)).to(self.config.device)
        self.word_context_weight = nn.Parameter(torch.Tensor(2 * self.word_hidden_size, 1)).to(self.config.device)

        self.sent_weight = nn.Parameter(torch.Tensor(2 * self.sent_hidden_size,
                                                     2 * self.sent_hidden_size)).to(self.config.device)
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * self.sent_hidden_size)).to(self.config.device)
        self.sent_context_weight = nn.Parameter(torch.Tensor(2 * self.sent_hidden_size, 1)).to(self.config.device)

        self.word_gru = nn.GRU(self.emb_d, self.word_hidden_size, batch_first=True,
                               num_layers=config.num_layers, bidirectional=True,
                               dropout=config.dropout).to(self.config.device)

        self.sent_gru = nn.GRU(2 * self.word_hidden_size, self.sent_hidden_size, batch_first=True,
                               num_layers=config.num_layers, bidirectional=True,
                               dropout=config.dropout).to(self.config.device)

        self.cls = nn.Linear(2 * self.sent_hidden_size, self.num_labels).to(self.config.device)

        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.word_context_weight.data.normal_(mean, std)
        self.sent_weight.data.normal_(mean, std)
        self.sent_context_weight.data.normal_(mean, std)

    def _init_hidden_state(self, last_batch_size):
        batch_size = last_batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size).to(self.device)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size).to(self.device)

    def _word_attn_net(self, sent_ids):
        # sent_ids shape (batch_size, word_size)
        embeds = self.emb(sent_ids)  # (batch_size, word_size, word_dim)
        f_out, _ = self.word_gru(embeds)  # (batch_size, word_size, word_hidden_size * 2)
        out = matrix_mul(f_out, self.word_weight, self.word_bias)  # (batch_size, word_size, word_hidden_size * 200)
        if len(out.size()) != 3:
            out = out.unsqueeze(0)
        out = matrix_mul(out, self.word_context_weight)  # (batch_size, word_size, 1)
        if len(out.size()) != 3:
            out = out.unsqueeze(0)
        attn_weight = F.softmax(out, dim=1)  # as above
        out = element_wise_mul(f_out, attn_weight)  # (batch_size, word_hidden_size)

        return out

    def _sent_attn_net(self, input_ids):
        # input_ids shape: (sent_size, batch_size, word_size)
        sent_list = [self._word_attn_net(sent_ids) for sent_ids in input_ids]
        out = torch.cat(sent_list, 0)
        out = out.permute(1, 0, 2)  # (batch_size, sent_size, word_hidden_size * 2)
        f_output, h_output = self.sent_gru(out)  # (batch_size, sent_size, sent_hidden_size * 2)
        out = matrix_mul(f_output, self.sent_weight, self.sent_bias)  # (batch_size, sent_size, sent_hidden_size * 2)
        if len(out.size()) != 3:
            out = out.unsqueeze(0)
        out = matrix_mul(out, self.sent_context_weight)  # (batch_size, sent_size, 1)
        if len(out.size()) != 3:
            out = out.unsqueeze(0)
        attn_weight = F.softmax(out, dim=1)  # as above
        out = element_wise_mul(f_output, attn_weight).squeeze()

        return out

    def forward(self, input_ids):
        input_ids = pad_seq_han(input_ids).to(self.config.device)
        input_ids = input_ids.permute(1, 0, 2)  # (max_sent_length, batch_size, max_word_length)

        out = self._sent_attn_net(input_ids)

        logits = self.cls(out)
        preds = torch.argmax(logits, dim=1)

        return logits, preds
