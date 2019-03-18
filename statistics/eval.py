# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/6/14 12:57
'''
import os

eval_path = '.'
eval_script = os.path.join(eval_path, "conlleval")
output_path = 'data/label/intelligence/all/decode_crf_raw_all_bio.txt'

os.system("%s < %s" % (eval_script, output_path))