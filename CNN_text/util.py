# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/10/14 11:31
'''

import re
import os
import random
import tarfile
import urllib
from torchtext import data


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class DATASET(TarDataset):

    # url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    # filename = 'rt-polaritydata.tar'
    dirname = 'data/conll' #CNN_text/

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            # with open(os.path.join(path, 'BioNLP13CG.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'BioNLP13CG'], fields) for line in f]
            # with open(os.path.join(path, 'BioNLP13PC.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'BioNLP13PC'], fields) for line in f]
            # with open(os.path.join(path, 'CRAFT.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'CRAFT'], fields) for line in f]
            with open(os.path.join(path, 'kaggle.train'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'kaggle'], fields) for line in f]
            # with open(os.path.join(path, 'BC5CDR.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'BC5CDR'], fields) for line in f]
            with open(os.path.join(path, 'conll2003.train'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'conll2003'], fields) for line in f]
            # with open(os.path.join(path, 'BioNLP13CG.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'BioNLP13CG'], fields) for line in f]
            # with open(os.path.join(path, 'BioNLP13PC.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'BioNLP13PC'], fields) for line in f]
            # with open(os.path.join(path, 'CRAFT.train'), errors='ignore') as f:
            #     examples += [
            #         data.Example.fromlist([line, 'CRAFT'], fields) for line in f]
        super(DATASET, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download_or_unzip(root)
        path = None
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))
