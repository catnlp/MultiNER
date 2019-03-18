# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/7/10 14:27
'''
from E_util.metric import get_ner_fmeasure
from E_module.multiner import MultiNER
from CNN_text import train

import time
import sys
import random
import copy
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import visdom

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

########################################################### CNN_text
# import datetime
# import torch
# import torchtext.data as data
# from CNN_text import model
# from CNN_text import train
# from CNN_text import util
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
#
# # learning
# class Args(object):
#     def __init__(self):
#         self.readme = 'for args'
#
# cnn_args = Args()
# cnn_args.lr = 0.001
# cnn_args.epochs = 256
# cnn_args.batch_size = 64
# cnn_args.log_interval = 1
# cnn_args.test_interval = 100
# cnn_args.save_interval = 500
# cnn_args.save_dir = 'snapshot'
# cnn_args.early_stop = 1000
# cnn_args.save_best = True
# cnn_args.shuffle = False
# cnn_args.dropout = 0.5
# cnn_args.max_norm = 3.0
# cnn_args.embed_dim = 128
# cnn_args.kernel_num = 100
# cnn_args.kernel_sizes = '3,4,5'
# cnn_args.static = False
# cnn_args.device = -1
# cnn_args.no_cuda = False
# cnn_args.snapshot = 'CNN_text/snapshot/conll-kaggle-2003/best_steps_17000.pt'
# cnn_args.predict = 'You just make me so sad and I have to leave you .'
# cnn_args.test = False
#
#
# def disease(text_field, label_field, **kargs):
#     train_data, dev_data = util.DATASET.splits(text_field, label_field)
#     text_field.build_vocab(train_data, dev_data)
#     label_field.build_vocab(train_data, dev_data)
#     train_iter, dev_iter = data.Iterator.splits(
#         (train_data, dev_data),
#         batch_sizes=(cnn_args.batch_size, len(dev_data)),
#         **kargs)
#     return train_iter, dev_iter
#
#
# # load data
# print("\nLoading data...")
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter = disease(text_field, label_field, device=-1, repeat=False)
# # train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# # train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)
#
#
# # update args and print
# cnn_args.embed_num = len(text_field.vocab)
# cnn_args.class_num = len(label_field.vocab) - 1
# cnn_args.cuda = (not cnn_args.no_cuda) and torch.cuda.is_available()
# del cnn_args.no_cuda
# cnn_args.kernel_sizes = [int(k) for k in cnn_args.kernel_sizes.split(',')]
# cnn_args.save_dir = os.path.join(cnn_args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#
# print("\nParameters:")
# for attr, value in sorted(cnn_args.__dict__.items()):
#     print("\t{}={}".format(attr.upper(), value))
#
# # model
# cnn = model.CNN_Text(cnn_args)
# if cnn_args.snapshot is not None:
#     print('\nLoading model from {}...'.format(cnn_args.snapshot))
#     cnn.load_state_dict(torch.load(cnn_args.snapshot))
#
# if cnn_args.cuda:
#     torch.cuda.set_device(cnn_args.device)
#     cnn = cnn.cuda()
#
# # train or predict
# if cnn_args.predict is not None:
#     # label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
#     # print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
#     score = train.score(cnn_args.predict, cnn, text_field, cnn_args.cuda)
#     print('\n[Text]  {}\n[Label] {}\n'.format(cnn_args.predict, score))

###########################################################

def data_initialization(data, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)

def predict_check(pred_variable, gold_variable, mask_variable):
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token

def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, ignore=False):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    # batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    tag_label = []
    dict = {}
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert(len(pred) == len(gold))
        if ignore:
            if (pred[0] != pred[-1]):
                print('Error datasets!')
                exit(1)
            tag = pred[0][2: ]
            if tag not in tag_label:
                dict[tag] = [[], []]
                tag_label.append(tag)
            dict[tag][0].append(pred[1: -1])
            dict[tag][1].append(gold[1: -1])
            pred_label.append(pred[1: -1])
            gold_label.append(gold[1: -1])
        else:
            pred_label.append(pred)
            gold_label.append(gold)

    return pred_label, gold_label, dict

def save_data_setting(data, save_file):
    new_data = copy.deepcopy(data)
    new_data.train_texts = []
    new_data.dev_texts = []
    new_data.test_texts = []
    new_data.raw_texts = []

    new_data.train_ids = []
    new_data.dev_ids = []
    new_data.test_ids = []
    new_data.raw_ids = []

    with open(save_file, 'wb+') as fp:
        pickle.dump(new_data, fp)
    print('Data setting saved to file: ', save_file)

def load_data_setting(save_file):
    with open(save_file, 'r') as fp:
        data = pickle.load(fp)
    print('Data setting loaded from file: ', save_file)
    data.show_data_summary()
    return data

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1+decay_rate*epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate(data, model, name, corpus_id):
    if name == 'train':
        instances = data.train_ids[corpus_id]
    elif name == 'dev':
        instances = data.dev_ids[corpus_id]
    elif name == 'test':
        instances = data.test_ids[corpus_id]
    elif name == 'raw':
        instances = data.raw_ids[corpus_id]
    else:
        instances = None
        print('Error: wrong evaluate name, ', name)

    pred_results = []
    gold_results = []

    model.eval()
    batch_size = data.batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id+1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start: end]
        if not instance:
            continue
        batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.gpu, True)
        tag_seq = model(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, corpus_id)
        pred_label, gold_label, dict_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label

    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)

    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    return speed, acc, p, r, f

def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    chars = [sent[1] for sent in input_batch_list]
    labels = [sent[2] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    for idx, (seq, label,seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        if seqlen == 0:
            print(words)
        word_seq_tensor[idx, : seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, : seqlen] = torch.LongTensor(label)
        mask[idx, : seqlen] = torch.Tensor([1]*seqlen)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars] #catnlp
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)), volatile=volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)

    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask

def train_model(data, name, save_dset, save_model_dir, seg=True):
    print('---Training model---')
    data.show_data_summary()
    save_data_name = save_dset
    save_data_setting(data, save_data_name)
    model = MultiNER(data)
    if data.gpu:
        model = model.cuda()


    if data.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif data.optim.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters())
    elif data.optim.lower() == 'adadelta':
        optimizer = optim.Adadelta(model.parameters())
    elif data.optim.lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters())
    elif data.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum)
    else:
        optimizer = None
        print('Error optimizer selection, please check config.optim.')
        exit(1)

    num_corpus = data.num_corpus
    epoch = data.iteration
    vis = visdom.Visdom()
    losses = [[] for _ in range(num_corpus)]
    all_F = [[[0., 0., 0.]] for _ in range(num_corpus)]
    scores = [[] for _ in range(num_corpus)]

    for idx in range(epoch):
        epoch_start = time.time()
        print('Epoch: %s/%s' % (idx, epoch))
        step_size = data.step_size
        batch_size = []
        sample_loss = []
        total_loss = []
        right_token = []
        whole_token = []
        for k in range(num_corpus):
            batch_size.append(len(data.train_ids[k]) // step_size)
            random.shuffle(data.train_ids[k])
            sample_loss.append(0)
            total_loss.append(0)
            right_token.append(0)
            whole_token.append(0)

        for step_id in range(1, step_size):
            for corpus_id in range(num_corpus):
                if data.optim.lower() == 'sgd':
                    optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)
                model.train()
                model.zero_grad()
                start = (step_id - 1) * batch_size[corpus_id]
                end = (step_id) * batch_size[corpus_id]
                instance = data.train_ids[corpus_id][start: end]
                if not instance:
                    continue

########################################################################
                tag = True
                # #if corpus_id != 0:
                # score = 0.0
                # count = 0
                # for sent in instance:
                #     line = ""
                #     count += 1
                #     if(len(sent[0]) <= 5):
                #         continue
                #     for word in sent[0]:
                #         line += data.word_alphabet.get_instance(word) + " "
                #     # print(line)
                #     score_tmp = train.score(line, cnn, text_field, cnn_args.cuda)
                #     # print('\n[Text]  {}\n[Label] {}\n'.format(line, score_tmp))
                #     score += score_tmp[0][1].data[0]
                #
                # if count > 0:
                #     score /= count
                # # print('\n[Score] {}\n'.format(score))
                # # if corpus_id == 0 and score.data[0] < 0.16:
                # #     tag = False
                #
                # # if corpus_id == 0 and score < 0.06:
                # #     tag = False
                #
                # if corpus_id == 0 and score < 0.03:
                #     tag = False
                #
                # # if corpus_id == 2 and score < 0.25:
                # #     tag = False
                # #
                # # if corpus_id == 3 and score < 0.16:
                # #     tag = False
                # #
                # # if corpus_id == 4 and score < 0.25:
                # #     tag = False
                #
                # sys.stdout.flush()
                # # scores[corpus_id].append(score)
                # # Lwin = 'Score of ' + name + '_' + str(corpus_id)
                # # vis.line(np.array(scores[corpus_id]), X=np.array([i for i in range(len(scores[corpus_id]))]), win=Lwin, opts={'title': Lwin, 'legend': ['score']})

#######################################################################################################

                batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.gpu)
                loss, tag_seq = model.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask, corpus_id, tag)
                right, whole = predict_check(tag_seq, batch_label, mask)
                right_token[corpus_id] += right
                whole_token[corpus_id] += whole
                sample_loss[corpus_id] += loss.data[0]

                if step_id % 50 == 0:
                    print('\tStep: %s; corpus: %s; loss: %.4f; acc: %s/%s=%.4f'
                            % ( step_id, corpus_id, sample_loss[corpus_id], right_token[corpus_id], whole_token[corpus_id], (right_token[corpus_id]+0.0) / whole_token[corpus_id]))
                    sys.stdout.flush()
                    losses[corpus_id].append(sample_loss[corpus_id] / 50.0)
                    Lwin = 'Loss of ' + name + '_' + str(corpus_id)
                    vis.line(np.array(losses[corpus_id]), X=np.array([i for i in range(len(losses[corpus_id]))]),
                                win=Lwin, opts={'title': Lwin, 'legend': ['loss']})
                    sample_loss[corpus_id] = 0

                loss.backward()
                if data.clip:
                    torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
                optimizer.step()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print('Epoch: %s training finished. Time: %.2fs' % (idx, epoch_cost))
        print('Epoch: %s test start' % idx)
        for corpus_id in range(num_corpus):
            print('\tCorpus: ', corpus_id)
            speed, acc, p, r, f_train = evaluate(data, model, 'train', corpus_id)
            speed, acc, p, r, f_dev = evaluate(data, model, 'dev', corpus_id)
            speed, acc, p, r, f_test = evaluate(data, model, 'test', corpus_id)
            print('\t\tF1 score is %.4f' % f_test)

            all_F[corpus_id].append([f_train * 100.0, f_dev * 100.0, f_test * 100.0])
            Fwin = 'F1-score of ' + name + '_' + str(corpus_id) + ' {train, dev, test}'
            vis.line(np.array(all_F[corpus_id]), X=np.array([i for i in range(len(all_F[corpus_id]))]),
                     win=Fwin, opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})

        test_finish = time.time()
        test_cost = test_finish - epoch_finish

        print('Epoch: %s testing finished. Test: %.2fs' % (idx, test_cost))
        # if current_score > best_test:
        #     if seg:
        #         print('Exceed previous best f score: ', best_test)
        #     else:
        #         print('Exceed previous best acc score: ', best_test)
        model_name = save_model_dir + '/' + name
        torch.save(model.state_dict(), model_name)
        # best_test = current_score
        # with open(save_model_dir + '/' + name + '_eval_' + str(idx) + '.txt', 'w') as f:
        #     if seg:
        #         f.write('acc: %.4f, p: %.4f, r: %.4f, f: %.4f' % (acc, p, r, best_test))
        #         f.write('acc: %.4f, p: %.4f' % (acc, p))
        #     else:
        #         f.write('acc: %.4f' % acc)
        #
        # if seg:
        #     print('Current best f score: ', best_test)
        # else:
        #     print('Current best acc score: ', best_test)

        gc.collect()

# def load_model_decode(model_dir, data, name, gpu, seg=True):
#     data.gpu = gpu
#     print('Load Model from file: ', model_dir)
#     model = NER(data)
#     model.load_state_dict(torch.load(model_dir))
#     print('Decode % data ... ' % name)
#     start_time = time.time()
#     speed, acc, p, r, f, pred_results = evaluate(data, model, name)
#     end_time = time.time()
#     time_cost = end_time - start_time
#     if seg:
#         print('%s: time: %.2f, speed:%.2ft/s; acc: %.4f, r: %.4f, f: %.4f'
#               % (name, time_cost, speed, acc, p, r, f))
#     else:
#         print('%s: time: %.2f, speed: %.2ft/s; acc: %.4f' % (name, time_cost, speed, acc))
#     return pred_results

# def test_optimizer(data):
#     print('---Test Optimizers---')
#     model_SGD = NER(data)
#     model_Adam = NER(data)
#     model_RMSprop = NER(data)
#     model_Adadelta = NER(data)
#     model_Adagrad = NER(data)
#
#     if data.gpu:
#         model_SGD = model_SGD.cuda()
#         model_Adam = model_Adam.cuda()
#         model_RMSprop = model_RMSprop.cuda()
#         model_Adadelta = model_Adadelta.cuda()
#         model_Adagrad = model_Adagrad.cuda()
#
#     optimizer_SGD = optim.SGD(model_SGD.parameters(), lr=data.lr, momentum=data.momentum)
#     optimizer_Adam = optim.Adam(model_Adam.parameters())
#     optimizer_RMSprop = optim.RMSprop(model_RMSprop.parameters())
#     optimizer_Adadelta = optim.Adadelta(model_Adadelta.parameters())
#     optimizer_Adagrad = optim.Adagrad(model_Adagrad.parameters())
#
#     epoch = data.iteration
#     vis = visdom.Visdom()
#     losses = []
#     train_F = [[0., 0., 0., 0., 0.]]
#     dev_F = [[0., 0., 0., 0., 0.]]
#     test_F = [[0., 0., 0., 0., 0.]]
#     for idx in range(epoch):
#         epoch_start = time.time()
#         print('Epoch: %s/%s' % (idx, epoch))
#
#         optimizer_SGD = lr_decay(optimizer_SGD, idx, data.lr_decay, data.lr)
#         instance_count = 0
#         sample_loss_SGD = 0
#         sample_loss_Adam = 0
#         sample_loss_RMSprop = 0
#         sample_loss_Adadelta = 0
#         sample_loss_Adagrad = 0
#         random.shuffle(data.train_ids)
#
#         model_SGD.train()
#         model_Adam.train()
#         model_RMSprop.train()
#         model_Adadelta.train()
#         model_Adagrad.train()
#         model_SGD.zero_grad()
#         model_Adam.zero_grad()
#         model_RMSprop.zero_grad()
#         model_Adadelta.zero_grad()
#         model_Adagrad.zero_grad()
#
#         batch_size = data.batch_size
#         train_num = len(data.train_ids)
#         total_batch = train_num // batch_size + 1
#         for batch_id in range(total_batch):
#             start = batch_id * batch_size
#             end = (batch_id+1) * batch_size
#             if end > train_num:
#                 end = train_num
#             instance = data.train_ids[start: end]
#             if not instance:
#                 continue
#             batch_word, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.gpu)
#             instance_count += 1
#             loss_SGD, tag_seq_SGD = model_SGD.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
#             loss_Adam, tag_seq_Adam = model_Adam.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
#             loss_RMSprop, tag_seq_RMSprop = model_RMSprop.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
#             loss_Adadelta, tag_seq_Adadelta = model_Adadelta.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
#             loss_Adagrad, tag_seq_Adagrad = model_Adagrad.neg_log_likelihood_loss(batch_word, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
#
#             sample_loss_SGD += loss_SGD.data[0]
#             sample_loss_Adam += loss_Adam.data[0]
#             sample_loss_RMSprop += loss_RMSprop.data[0]
#             sample_loss_Adadelta += loss_Adadelta.data[0]
#             sample_loss_Adagrad += loss_Adagrad.data[0]
#
#             if end % 500 == 0:
#                 sys.stdout.flush()
#                 losses.append([sample_loss_SGD/50.0, sample_loss_Adam/50.0, sample_loss_RMSprop/50.0, sample_loss_Adadelta/50.0, sample_loss_Adagrad/50.0])
#                 Lwin = 'Loss of Optimizers'
#                 vis.line(np.array(losses), X=np.array([i for i in range(len(losses))]),
#                          win=Lwin, opts={'title': Lwin, 'legend': ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad']})
#                 sample_loss_SGD = 0
#                 sample_loss_Adam = 0
#                 sample_loss_RMSprop = 0
#                 sample_loss_Adadelta = 0
#                 sample_loss_Adagrad = 0
#             loss_SGD.backward()
#             loss_Adam.backward()
#             loss_RMSprop.backward()
#             loss_Adadelta.backward()
#             loss_Adagrad.backward()
#             # if data.clip:
#             #     torch.nn.E_util.clip_grad_norm(model.parameters(), 10.0)
#             optimizer_SGD.step()
#             optimizer_Adam.step()
#             optimizer_RMSprop.step()
#             optimizer_Adadelta.step()
#             optimizer_Adagrad.step()
#             model_SGD.zero_grad()
#             model_Adam.zero_grad()
#             model_RMSprop.zero_grad()
#             model_Adadelta.zero_grad()
#             model_Adagrad.zero_grad()
#
#         epoch_finish = time.time()
#         epoch_cost = epoch_finish - epoch_start
#         print('Epoch: %s training finished. Time: %.2fs, speed: %.2ft/s' % (idx, epoch_cost, train_num/epoch_cost))
#
#         speed, acc, p, r, f_train_SGD, _ = evaluate(data, model_SGD, 'train')
#         speed, acc, p, r, f_train_Adam, _ = evaluate(data, model_Adam, 'train')
#         speed, acc, p, r, f_train_RMSprop, _ = evaluate(data, model_RMSprop, 'train')
#         speed, acc, p, r, f_train_Adadelta, _ = evaluate(data, model_Adadelta, 'train')
#         speed, acc, p, r, f_train_Adagrad, _ = evaluate(data, model_Adagrad, 'train')
#
#         train_F.append([f_train_SGD*100, f_train_Adam*100, f_train_RMSprop*100, f_train_Adadelta*100, f_train_Adagrad*100])
#         train_Fwin = 'F1-score of Optimizers{train}'
#         vis.line(np.array(train_F), X=np.array([i for i in range(len(train_F))]),
#                  win=train_Fwin, opts={'title': train_Fwin, 'legend': ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad']})
#
#         speed, acc, p, r, f_dev_SGD, _ = evaluate(data, model_SGD, 'dev')
#         speed, acc, p, r, f_dev_Adam, _ = evaluate(data, model_Adam, 'dev')
#         speed, acc, p, r, f_dev_RMSprop, _ = evaluate(data, model_RMSprop, 'dev')
#         speed, acc, p, r, f_dev_Adadelta, _ = evaluate(data, model_Adadelta, 'dev')
#         speed, acc, p, r, f_dev_Adagrad, _ = evaluate(data, model_Adagrad, 'dev')
#
#         dev_F.append([f_dev_SGD * 100, f_dev_Adam * 100, f_dev_RMSprop * 100, f_dev_Adadelta * 100,
#                         f_dev_Adagrad * 100])
#         dev_Fwin = 'F1-score of Optimizers{dev}'
#         vis.line(np.array(dev_F), X=np.array([i for i in range(len(dev_F))]),
#                  win=dev_Fwin, opts={'title': dev_Fwin, 'legend': ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad']})
#
#         speed, acc, p, r, f_test_SGD, _ = evaluate(data, model_SGD, 'test')
#         speed, acc, p, r, f_test_Adam, _ = evaluate(data, model_Adam, 'test')
#         speed, acc, p, r, f_test_RMSprop, _ = evaluate(data, model_RMSprop, 'test')
#         speed, acc, p, r, f_test_Adadelta, _ = evaluate(data, model_Adadelta, 'test')
#         speed, acc, p, r, f_test_Adagrad, _ = evaluate(data, model_Adagrad, 'test')
#
#         test_F.append([f_test_SGD * 100, f_test_Adam * 100, f_test_RMSprop * 100, f_test_Adadelta * 100,
#                       f_test_Adagrad * 100])
#         test_Fwin = 'F1-score of Optimizers{test}'
#         vis.line(np.array(test_F), X=np.array([i for i in range(len(test_F))]),
#                  win=test_Fwin, opts={'title': test_Fwin, 'legend': ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad']})
#         gc.collect()