# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/6/5 12:55
'''
from C_util.config import Config
from C_util.helpers import *

import sys
import argparse
import random
import torch
import numpy as np

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NER')
    parser.add_argument('--wordemb', help='Embedding for words', default='zhwiki')
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default='C_model/intelligence/train_all_bio') # catnlp
    parser.add_argument('--savedset', help='Dir of saved data setting', default='C_model/intelligence/train_all_bio.dset') # catnlp
    parser.add_argument('--train', default='data/C_group/intelligence/train-all.bio')
    parser.add_argument('--test', default='data/C_group/intelligence/test.bio')
    parser.add_argument('--gpu', default='True')
    parser.add_argument('--seg', default='True')
    parser.add_argument('--extendalphabet', default='True')
    parser.add_argument('--raw', default='data/C_group/intelligence/test.bio')
    parser.add_argument('--loadmodel', default='C_model/intelligence/train_all_bio/intelligence_train_all_bio')
    parser.add_argument('--output', default='data/C_group/intelligence/decode_train_all_bio.txt')
    args = parser.parse_args()

    train_file = args.train
    # dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == 'true':
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    if args.gpu.lower() == 'false':
        gpu = False
    else:
        gpu = torch.cuda.is_available()

    print('Seed num: ', seed_num)
    print('GPU available: ', gpu)
    print('Status: ', status)

    print('Seg: ', seg)
    print('Train file: ', train_file)
    # print('Dev file: ', dev_file)
    print('Test file: ', test_file)
    print('Raw file: ', raw_file)
    if status == 'train':
        print('Model saved to: ', save_model_dir)
    sys.stdout.flush()

    if status == 'train':
        emb = args.wordemb.lower()
        print('Word Embedding: ', emb)
        if emb == 'wiki':
            emb_file = 'data/embedding/wiki_100.utf8'
        elif emb == 'zhwiki':
            emb_file = 'data/embedding/zhwiki_100.txt'
        elif emb == 'sogou':
            emb_file = 'data/embedding/sogou_news_100.txt'
        else:
            emb_file = None
        # char_emb_file = args.charemb.lower()
        # print('Char Embedding: ', char_emb_file)

        name = 'BaseLSTM'  # catnlp
        config = Config()
        config.optim = 'Adam'
        config.lr = 0.015
        config.hidden_dim = 200
        config.bid_flag = True
        config.number_normalized = False
        data_initialization(config, train_file, test_file)
        config.gpu = gpu
        config.word_features = name
        print('Word features: ', config.word_features)
        config.generate_instance(train_file, 'train')
        # config.generate_instance(dev_file, 'dev')
        config.generate_instance(test_file, 'test')
        if emb_file:
            print('load word emb file...norm: ', config.norm_word_emb)
            config.build_word_pretain_emb(emb_file)
        # if char_emb_file != 'none':
        #     print('load char emb file...norm: ', config.norm_char_emb)
        #     config.build_char_pretrain_emb(char_emb_file)
        name = 'intelligence_train_all_bio'
        train(config, name, dset_dir, save_model_dir, seg)
    elif status == 'test':
        data = load_data_setting(dset_dir)
        # data.generate_instance(dev_file, 'dev')
        # load_model_decode(model_dir, data, 'dev', gpu, seg)
        data.generate_instance(test_file, 'test')
        load_model_decode(model_dir, data, 'test', gpu, seg)
    elif status == 'decode':
        data = load_data_setting(dset_dir)
        data.generate_instance(raw_file, 'raw')
        decode_results, gold_results  = load_model_decode(model_dir, data, 'raw', gpu, seg)
        data.write_decoded_results(output_file, gold_results, decode_results, 'raw')
    else:
        print('Invalid argument! Please use valid arguments! (train/test/decode)')