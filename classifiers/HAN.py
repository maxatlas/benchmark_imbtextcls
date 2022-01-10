import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import TaskModel, pad_seq_han
from utils import element_wise_mul, matrix_mul


class Model(TaskModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
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

        self.word_gru = nn.GRU(self.emb_d, self.word_hidden_size, num_layers=config.num_layers,
                               bidirectional=True, dropout=config.dropout).to(self.config.device)
        self.sent_gru = nn.GRU(2 * self.word_hidden_size, self.sent_hidden_size, num_layers=config.num_layers,
                               bidirectional=True, dropout=config.dropout).to(self.config.device)
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

    def forward(self, input_ids):
        sent_list = []

        input_ids = pad_seq_han(input_ids).to(self.config.device)
        input_ids = input_ids.permute(1, 0, 2)  # (max_sent_length, batch_size, max_word_length)

        for sent in input_ids:
            sent = sent.permute(1, 0)  # (max_word_length, batch) nth word from each batch
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
        preds = torch.argmax(logits, dim=1)

        return logits, preds
