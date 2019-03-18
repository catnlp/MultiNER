# encoding:utf-8
'''
@Author: catnlp
@Email: wk_nlp@163.com
@Time: 2018/6/15 16:00
'''

class Analysis():
    def __init__(self, path):
        if path[-1] != '/':
            path += '/'
        self.path = path

    def countNewWords(self, output, new, dict):
        output_path = self.path + output
        new_path = self.path + new
        dict_path = self.path + dict

        dict_set = set()
        with open(dict_path) as dict:
            lines = dict.readlines()
            for line in lines:
                line = line.replace('\n', '')
                if not line:
                    continue
                word = line.split('\t')
                dict_set.add(word[0])

        with open(output_path) as output, open(new_path, 'w') as new:
            lines = output.readlines()
            str = ''
            new_dict = {}
            count = 0
            tag = ''
            num = 0
            for line in lines:
                num += 1
                line = line.replace('\n', '')
                if not line:
                    continue
                words = line.split('\t')
                if len(words) < 3:
                    continue
                if words[2] != 'O':
                    # print(words[2])
                    if words[2][0] == 'B':
                        if str != '' and str not in dict_set:
                            print(num)
                            print(str)
                            count += 1
                            new_dict[str] = tag
                        str = ''
                        tag = words[2][2: ]
                    str += words[0]

                    # print(words[0] + '\t' + words[2])
                else:
                    if str != '' and str not in dict_set:
                        print(num)
                        print(str)
                        count += 1
                        new_dict[str] = tag
                    str = ''
            for key in new_dict:
                # print(key)
                # print(str)
                new.write(key + '\t' + new_dict[key] +'\n')
            print(count)

if __name__ == '__main__':
    analysis = Analysis('data/label/intelligence/all')
    analysis.countNewWords('decode_train_all_bio.txt', 'newwords_train_all_bio.txt', 'dictAll.txt')