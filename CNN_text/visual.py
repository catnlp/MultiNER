# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/10/29 14:24
'''

import visdom
vis = visdom.Visdom()

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets, manifold

import torch.autograd as autograd

# digits = datasets.load_digits(n_class=10)
# X, y = digits.data, digits.target
# print(X.shape)
# print(X[0])
# print(y.shape)
# n_samples, n_features = X.shape

########################################################### CNN_text
import datetime
import torch
import torchtext.data as data
from CNN_text import model
from CNN_text import train
from CNN_text import util

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# learning
class Args(object):
    def __init__(self):
        self.readme = 'for args'

cnn_args = Args()
cnn_args.lr = 0.001
cnn_args.epochs = 256
cnn_args.batch_size = 64
cnn_args.log_interval = 1
cnn_args.test_interval = 100
cnn_args.save_interval = 500
cnn_args.save_dir = 'snapshot'
cnn_args.early_stop = 1000
cnn_args.save_best = True
cnn_args.shuffle = False
cnn_args.dropout = 0.5
cnn_args.max_norm = 3.0
cnn_args.embed_dim = 128
cnn_args.kernel_num = 100
cnn_args.kernel_sizes = '3,4,5'
cnn_args.static = False
cnn_args.device = -1
cnn_args.no_cuda = False
cnn_args.snapshot = 'snapshot/conll-kaggle-2003/best_steps_17000.pt'
cnn_args.predict = 'You just make me so sad and I have to leave you .'
cnn_args.test = False


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
print(cnn_args.embed_num)
cnn_args.class_num = len(label_field.vocab) - 1
cnn_args.cuda = (not cnn_args.no_cuda) and torch.cuda.is_available()
del cnn_args.no_cuda
cnn_args.kernel_sizes = [int(k) for k in cnn_args.kernel_sizes.split(',')]
cnn_args.save_dir = os.path.join(cnn_args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) # datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

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
all_text = []
all_label = []

with open("data/conll/kaggle.train") as rf:
    content = ""
    for line in rf:
        line = line.replace("\n", "")
        if not line:
            continue
        text = text_field.preprocess(line)
        if (len(text) <= 5):
            continue
        text = [[text_field.vocab.stoi[x] for x in text]]
        x = text_field.tensor_type(text)
        x = autograd.Variable(x, volatile=True)
        if cnn_args.cuda:
            x = x.cuda()
        output = cnn.forward_emb(x)
        # output = cnn.forward(x)
        # output = x
        all_text.append(output)
        all_label.append(0)

with open("data/conll/conll2003.train") as rf:
    content = ""
    for line in rf:
        line = line.replace("\n", "")
        if not line:
            continue
        text = text_field.preprocess(line)
        if (len(text) <= 5):
            continue
        text = [[text_field.vocab.stoi[x] for x in text]]
        x = text_field.tensor_type(text)
        x = autograd.Variable(x, volatile=True)
        if cnn_args.cuda:
            x = x.cuda()
        output = cnn.forward_emb(x)
        # output = cnn.forward(x)
        # output = x
        all_text.append(output)
        all_label.append(1)

# with open("data/cc/CRAFT.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         # output = cnn.forward_emb(x)
#         output = cnn.forward(x)
#         # output = x
#         all_text.append(output)
#         all_label.append(2)

# with open("data/cc/BioNLP13CG.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if  cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(0)

# with open("data/cc/BioNLP13PC.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(1)
#
# with open("data/cc/CRAFT.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(2)

# with open("data/chem/BC4CHEMD.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(0)
#
# with open("data/chem/BC5CDR.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(1)
#
# with open("data/chem/BioNLP11ID.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(2)
#
# with open("data/chem/BioNLP13CG.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(3)
#
# with open("data/chem/BioNLP13PC.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(4)
#
# with open("data/cc/CRAFT.train") as rf:
#     content = ""
#     for line in rf:
#         line = line.replace("\n", "")
#         if not line:
#             continue
#         text = text_field.preprocess(line)
#         if (len(text) <= 5):
#             continue
#         text = [[text_field.vocab.stoi[x] for x in text]]
#         x = text_field.tensor_type(text)
#         x = autograd.Variable(x, volatile=True)
#         if cnn_args.cuda:
#             x = x.cuda()
#         output = cnn.forward_emb(x)
#         all_text.append(output)
#         all_label.append(5)

print(len(all_text))
print(len(all_text[0]))

print(all_text[:2])

# X = np.array(all_text)
# # n_samples, n_features = X.shape
# y = np.array(all_label)

word_seq_lengths = torch.LongTensor(list(map(len, all_text)))
max_seq_len = 300
X = torch.IntTensor(len(all_text), max_seq_len).zero_()
y = torch.IntTensor(len(all_label)).zero_() # np.array(all_label)



for idx, (seq, label, seqlen) in enumerate(zip(all_text,all_label, word_seq_lengths)):
    # if max_seq_len < seqlen:
    #     seqlen = max_seq_len
    # X[idx, :max_seq_len] = torch.from_numpy(seq.cpu().data.numpy())
    print(X[0].shape)
    print(seq.shape)
    seq_tmp = torch.from_numpy(seq.cpu().data.numpy())[0]
    X[idx, : len(seq_tmp)] = seq_tmp
    y[idx] = label

print(X.shape)
##################################################################################

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 2),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
t0 = time()
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
            "conll-kaggle-2003-1")

# plt.show()
vis.matplot(plt)