# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/7/10 17:28
'''
from E_module.char import Char

import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

class Embedding_layer(nn.Module):
    def __init__(self, config):
        super(Embedding_layer, self).__init__()
        print('---build batched Embedding layer---')
        self.gpu = config.gpu
        self.bidirectional = config.bid_flag
        self.batch_size = config.batch_size
        self.char_hidden_dim = 0

        self.use_char = config.use_char
        if self.use_char:
            self.char_hidden_dim = config.char_hidden_dim
            self.char_embedding_dim = config.char_emb_dim
            self.char = Char(config.char_features, config.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, config.dropout, self.gpu)

        self.embedding_dim = config.word_emb_dim
        self.hidden_dim = config.hidden_dim
        self.hyper_hidden_dim = config.hyper_hidden_dim
        self.hyper_embedding_dim = config.hyper_embedding_dim
        self.layers = config.layers
        self.drop = nn.Dropout(config.dropout)
        self.word_embeddings = nn.Embedding(config.word_alphabet.size(), self.embedding_dim)
        if config.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(config.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(config.word_alphabet.size(), self.embedding_dim)))

    def random_embedding(self, vocab_size, embedding_dim):
        initial_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            initial_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return initial_emb

    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        batch_size = word_inputs.size(0)
        sent_len = word_inputs.size(1)
        word_embs = self.word_embeddings(word_inputs)

        if self.use_char:
            char_features = self.char.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size, sent_len, -1)
            word_embs = torch.cat([word_embs, char_features], 2)

        word_embs = self.drop(word_embs)
        # if self.mode.startswith('Base'):
        #     packed_words = pack_padded_sequence(word_embs, word_seq_lengths.cpu().numpy(), True)
        #     out, _ = self.encoder(packed_words)
        #     out, _ = pad_packed_sequence(out)
        #     out = out.transpose(1, 0)
        # else:
        #     out, _ = self.encoder(word_embs)
        # out = self.drop(out) ## catnlp
        return word_embs

    # def get_output_score(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
    #     out = self.get_word_features(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
    #     return out
    #
    # def neg_log_likelihood_loss(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, batch_label):
    #     batch_size = word_inputs.size(0)
    #     seq_len = word_inputs.size(1)
    #     total_words = batch_size * seq_len
    #     loss_function = nn.NLLLoss(ignore_index=0, size_average=False)
    #     out = self.get_output_score(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
    #     out = out.view(total_words, -1)
    #     score = F.log_softmax(out, 1)
    #     loss = loss_function(score, batch_label.view(total_words))
    #     _, tag_seq = torch.max(score, 1)
    #     tag_seq = tag_seq.view(batch_size, seq_len)
    #     return loss, tag_seq
    #
    # def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask):
    #     batch_size = word_inputs.size(0)
    #     seq_len = word_inputs.size(1)
    #     total_words = batch_size * seq_len
    #     out = self.get_output_score(word_inputs, word_seq_lengths, char_seq_lengths, char_seq_recover)
    #     out = out.view(total_words, -1)
    #     _, tag_seq = torch.max(out, 1)
    #     tag_seq = tag_seq.view(batch_size, seq_len)
    #     decode_seq = mask.long() * tag_seq
    #     return decode_seq
