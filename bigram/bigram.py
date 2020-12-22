import numpy as np
from collections import defaultdict
import unicodedata
import sys


class Bigram:
    @staticmethod
    def processed_file(input_file):
        file = open(input_file, 'r')
        lis = []
        for line in file.readlines():
            lis.append(line.lower())
        file.close()

        result_list = []
        punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        for line in lis:
            if not line:
                continue

            new_message = ""
            if line[-1] == '\n':
                new_message = line[:-2]
            else:
                new_message = line

            word_list = new_message.translate(punctuation).split(' ')
            result = [word for word in word_list if word != '']
            result_list.append(result)
        return result_list

    @staticmethod
    def processed_single_message(input_message):
        if not input_message:
            return []

        new_message = ""
        if input_message[-1] == '\n':
            new_message = input_message[:-2]
        else:
            new_message = input_message

        punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        word_list = new_message.translate(punctuation).split(' ')
        result = [word for word in word_list if word != '']
        return result

    @staticmethod
    def bigram_probability_dict(input_file, k):
        input_list = Bigram.processed_file(input_file)
        word_count = defaultdict(int)
        for message in input_list:
            for word in message:
                word_count[word] += 1

        unkown_words = set()
        for word in word_count:
            if word_count[word] == 1:
                unkown_words.add(word)

        words_num = defaultdict(int)
        pair_num = defaultdict(int)
        for message in input_list:
            if not message:
                continue

            for i in range(len(message)):
                if message[i] in unkown_words:
                    message[i] = "unk"
                words_num[message[i]] += 1
                if i == 0:
                    words_num["s/"] += 1
                    pair_num["s/_" + message[i]] += 1
                elif i == len(message) - 1:
                    words_num["/e"] += 1
                    pair_num[message[i - 1] + "_" + message[i]] += 1
                    pair_num[message[i] + "_/e"] += 1
                else:
                    pair_num[message[i - 1] + "_" + message[i]] += 1

        all_words = words_num.keys()
        for word1 in all_words:
            for word2 in all_words:
                pair_num[word1 + "_" + word2] += k

        pro_dict = defaultdict(float)
        for pair, num in pair_num.items():
            words = pair.split('_')
            first_word = words[0]
            pro_dict[pair] = num / (words_num[first_word] + k * len(all_words))

        return pro_dict

    @staticmethod
    def perplexity(pro_dict, valid_list):
        total_number = 0
        pp = 0
        for message in valid_list:
            if not message:
                continue
            for i in range(len(message) + 1):
                total_number += 1
                if i == 0:
                    if pro_dict["s/_" + message[i]] != 0:
                        pp += -np.log(pro_dict["s/_" + message[i]])
                    else:
                        pp += -np.log(pro_dict["s/_" + "unk"])
                elif i == len(message):
                    if pro_dict[message[i - 1] + "_/e"] != 0:
                        pp += -np.log(pro_dict[message[i - 1] + "_/e"])
                    else:
                        pp += -np.log(pro_dict["unk" + "_/e"])
                else:
                    if pro_dict[message[i - 1] + "_" + message[i]] != 0:
                        pp += -np.log(pro_dict[message[i - 1] + "_" + message[i]])
                    elif pro_dict["unk" + "_" + message[i]] != 0:
                        pp += -np.log(pro_dict["unk" + "_" + message[i]])
                    elif pro_dict[message[i - 1] + "_" + "unk"] != 0:
                        pp += -np.log(pro_dict[message[i - 1] + "_" + "unk"])
                    else:
                        pp += -np.log(pro_dict["unk" + "_" + "unk"])
        return np.exp(pp / total_number)


    @staticmethod
    def perplexity_single_message(pro_dict, message):
        if not message:
            return 0.0

        pp = 0
        for i in range(len(message) + 1):
            if i == 0:
                if pro_dict["s/_" + message[i]] != 0:
                    pp += -np.log(pro_dict["s/_" + message[i]])
                else:
                    pp += -np.log(pro_dict["s/_" + "unk"])
            elif i == len(message):
                if pro_dict[message[i - 1] + "_/e"] != 0:
                    pp += -np.log(pro_dict[message[i - 1] + "_/e"])
                else:
                    pp += -np.log(pro_dict["unk" + "_/e"])
            else:
                if pro_dict[message[i - 1] + "_" + message[i]] != 0:
                    pp += -np.log(pro_dict[message[i - 1] + "_" + message[i]])
                elif pro_dict["unk" + "_" + message[i]] != 0:
                    pp += -np.log(pro_dict["unk" + "_" + message[i]])
                elif pro_dict[message[i - 1] + "_" + "unk"] != 0:
                    pp += -np.log(pro_dict[message[i - 1] + "_" + "unk"])
                else:
                    pp += -np.log(pro_dict["unk" + "_" + "unk"])
        return pp / (len(message) + 1)