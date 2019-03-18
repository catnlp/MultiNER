# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/7/10 17:18
'''
from E_module.component.RNNs import RNN, LSTM
from E_module.component.MetaRNNs import MetaRNN, MetaLSTM
from E_module.embedding_layer import Embedding_layer
from E_module.crf import CRF
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

class MultiNER(nn.Module):
    def __init__(self, config):
        super(MultiNER, self).__init__()
        print('---build batched MultiNER---')
        label_size = config.label_alphabet_size
        config.label_alphabet_size += 2

        self.drop = nn.Dropout(config.dropout)
        self.embedding = Embedding_layer(config)

        self.mode = config.word_features
        if self.mode == 'BaseRNN':
            self.shared = nn.RNN(config.char_hidden_dim + config.word_emb_dim, config.hidden_dim, num_layers=config.layers,
                                  batch_first=True)
        elif self.mode == 'RNN':
            self.shared = RNN(config.char_hidden_dim + config.word_emb_dim, config.hidden_dim, num_layers=config.layers,
                               gpu=config.gpu)
        elif self.mode == 'MetaRNN':
            self.shared = MetaRNN(config.char_hidden_dim + config.word_emb_dim, config.hidden_dim, config.hyper_hidden_dim,
                                  config.hyper_embedding_dim, num_layers=config.layers, gpu=config.gpu)
        elif self.mode == 'BaseLSTM':
            self.shared = nn.LSTM(config.char_hidden_dim + config.word_emb_dim, config.hidden_dim // 2,
                                   num_layers=config.layers, batch_first=True, bidirectional=True)
        elif self.mode == 'LSTM':
            self.shared = LSTM(config.char_hidden_dim + config.word_emb_dim, config.hidden_dim // 2, num_layers=config.layers,
                                gpu=config.gpu, bidirectional=config.bidirectional)
        elif self.mode == 'MetaLSTM':
            self.shared = MetaLSTM(config.char_hidden_dim + config.word_emb_dim, config.hidden_dim // 2,
                                   config.hyper_hidden_dim, config.hyper_embedding_dim, num_layers=config.layers,
                                    gpu=config.gpu, bidirectional=config.bidirectional)
        else:
            print('Error word feature selection, please check config.word_features.')
            exit(0)

        self.private = nn.ModuleList()
        self.hiden2tags = nn.ModuleList()
        self.crf = nn.ModuleList()
        for i in range(config.num_corpus):
            task_private = nn.LSTM(config.hidden_dim, config.hidden_dim // 2, batch_first=True, bidirectional=True)
            self.private.append(task_private)
            hidden2tag = nn.Linear(config.hidden_dim, label_size+2)
            self.hiden2tags.append(hidden2tag)
            self.crf.append(CRF(label_size, config.gpu))

        self.loss = nn.MSELoss()

        if config.gpu:
            self.embedding = self.embedding.cuda()
            self.shared = self.shared.cuda()
            for i in range(config.num_corpus):
                self.private[i] = self.private[i].cuda()
                self.hiden2tags[i] = self.hiden2tags[i].cuda()
                self.crf[i] = self.crf[i].cuda()

    def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label, mask, corpus_id, tag):
        # loss, tag_seq = self.encoder.neg_log_likelihood_loss(word_inputs, word_seq_lengths, batch_label)
        # return loss, tag_seq
        # outs = self.encoder.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # total_loss = self.crf.neg_log_likelihood_loss(outs, batch_label, mask)
        # scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        word_embs = self.embedding(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        if self.mode.startswith('Base'):
            packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
            out, _ = self.shared(packed_words)
            out, _ = pad_packed_sequence(out)
            out = out.transpose(1, 0)
        else:
            out, _ = self.shared(word_embs)
        if tag:
            out_id, _ = self.private[corpus_id](out)
        else:
            out_id, _ = self.private[corpus_id](out.detach())
        out_id = self.hiden2tags[corpus_id](out_id)

        total_loss = self.crf[corpus_id].neg_log_likelihood_loss(out_id, batch_label, mask)
        score, tag_seq = self.crf[corpus_id].viterbi_decode(out_id, mask)

        # if (corpus_id != 0):
        #     out0, _ = self.private[0](out)
        #     out0 = self.hiden2tags[0](out0)
        #     loss = self.loss(out_id, out0.detach())
        #     total_loss += loss

        return total_loss, tag_seq

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, corpus_id):
        # decode_seq = self.encoder(word_inputs, word_seq_lengths, mask)
        # return decode_seq
        # outs = self.encoder.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        # scores, tag_seq = self.crf.viterbi_decode(outs, mask)
        # return tag_seq
        word_embs = self.embedding(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        if self.mode.startswith('Base'):
            packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
            out, _ = self.shared(packed_words)
            out, _ = pad_packed_sequence(out)
            out = out.transpose(1, 0)
        else:
            out, _ = self.encoder(word_embs)
        out, _ = self.private[corpus_id](out)
        out = self.hiden2tags[corpus_id](out)
        score, tag_seq = self.crf[corpus_id].viterbi_decode(out, mask)
        return tag_seq

    # def get_word_features(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     return self.encoder.get_word_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)