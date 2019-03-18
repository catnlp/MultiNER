# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/11/13 20:20
'''

from E_util.tagSchemeConverter import *

BIO2BIOES("data/E_group/conll/kaggle/train.tsv", "data/E_group/conll/kaggle_bmes/train.tsv")
BIO2BIOES("data/E_group/conll/kaggle/devel.tsv", "data/E_group/conll/kaggle_bmes/devel.tsv")
BIO2BIOES("data/E_group/conll/kaggle/test.tsv", "data/E_group/conll/kaggle_bmes/test.tsv")