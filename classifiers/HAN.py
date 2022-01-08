import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import matrix_mul, element_wise_mul, get_label_ids


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
        super(HAN, self).__init__(config)
        self.num_labels = config.n_labels
        self.word_hidden_size = config.hidden_size
        self.sent_hidden_size = config.hidden_size

        self.word_hidden_state, self.sent_hidden_state = None, None

        self.word_weight = nn.Parameter(torch.Tensor(2 * self.word_hidden_size, 2 * self.word_hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * self.word_hidden_size))
        self.word_context_weight = nn.Parameter(torch.Tensor(2 * self.word_hidden_size, 1))

        self.sent_weight = nn.Parameter(torch.Tensor(2 * config.sent_hidden_size, 2 * config.sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * config.sent_hidden_size))
        self.sent_context_weight = nn.Parameter(torch.Tensor(2 * config.sent_hidden_size, 1))

        self.word_gru = nn.GRU(self.emb_d, self.word_hidden_size, num_layers=config.num_layers,
                               bidirectional=True, dropout=config.hidden_dropout_prob)
        self.sent_gru = nn.GRU(2 * self.word_hidden_size, self.sent_hidden_size, num_layers=config.num_layers,
                               bidirectional=True, dropout=config.hidden_dropout_prob)
        self.cls = nn.Linear(2 * self.sent_hidden_size, self.num_labels)

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

        input_ids = input_ids.permute(1, 0, 2) # (max_sent_length, batch_size, max_word_length)
        for sent in input_ids:
            sent = sent.permute(1, 0) # (max_word_length, batch) nth word from each batch
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

    def batch_train(self, input_ids, label_ids, loss_func, attention_mask=None, token_type_ids=None):
        logits, _ = self.forward(input_ids)
        loss = loss_func(logits, label_ids)

        return loss

    def batch_eval(self, input_ids, a, b, labels, label_names):
        with torch.no_grad():
            _, preds = self.forward(input_ids)
        return preds
