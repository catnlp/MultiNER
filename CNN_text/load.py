# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/10/15 15:45
'''

import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
from CNN_text import model
from CNN_text import train
from CNN_text import mydatasets
from CNN_text import util

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
class Args(object):
    def __init__(self):
        self.readme = 'for args'

cnn_args = Args()
# parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
cnn_args.lr = 0.001
# parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
cnn_args.epochs = 256
# parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
cnn_args.batch_size = 64
# parser.add_argument('-log-interval', type=int, default=1,
#                    help='how many steps to wait before logging training status [default: 1]')
cnn_args.log_interval = 1
# parser.add_argument('-test-interval', type=int, default=100,
#                    help='how many steps to wait before testing [default: 100]')
cnn_args.test_interval = 100
# parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
cnn_args.save_interval = 500
# parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
cnn_args.save_dir = 'snapshot'
# parser.add_argument('-early-stop', type=int, default=1000,
#                     help='iteration numbers to stop without performance increasing')
cnn_args.early_stop = 1000
# parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
cnn_args.save_best = True
# data
# parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
cnn_args.shuffle = False
# model
# parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
cnn_args.dropout = 0.5
# parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
cnn_args.max_norm = 3.0
# parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
cnn_args.embed_dim = 128
# parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
cnn_args.kernel_num = 100
# parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
#                     help='comma-separated kernel size to use for convolution')
cnn_args.kernel_sizes = '3,4,5'
# parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
cnn_args.static = False
# device
# parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
cnn_args.device = -1
# parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
cnn_args.no_cuda = False
# option
# parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
cnn_args.snapshot = './snapshot/2018-10-16_15-39-09/best_steps_39100.pt'
# parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
cnn_args.predict = 'You just make me so sad and I have to leave you .'
# parser.add_argument('-test', action='store_true', default=False, help='train or test')
cnn_args.test = False
# args = parser.parse_args()


def disease(text_field, label_field, **kargs):
    train_data, dev_data = util.DATASET.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(cnn_args.batch_size, len(dev_data)),
        **kargs)
    return train_iter, dev_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = disease(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
cnn_args.embed_num = len(text_field.vocab)
cnn_args.class_num = len(label_field.vocab) - 1
cnn_args.cuda = (not cnn_args.no_cuda) and torch.cuda.is_available()
del cnn_args.no_cuda
cnn_args.kernel_sizes = [int(k) for k in cnn_args.kernel_sizes.split(',')]
cnn_args.save_dir = os.path.join(cnn_args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(cnn_args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(cnn_args)
if cnn_args.snapshot is not None:
    print('\nLoading model from {}...'.format(cnn_args.snapshot))
    cnn.load_state_dict(torch.load(cnn_args.snapshot))

if cnn_args.cuda:
    torch.cuda.set_device(cnn_args.device)
    cnn = cnn.cuda()

# train or predict
if cnn_args.predict is not None:
    # label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    # print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    score = train.score(cnn_args.predict, cnn, text_field, cnn_args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(cnn_args.predict, score))


