import numpy as np
from collections import defaultdict
import unicodedata
import sys


class Trigram:
    @staticmethod
    def processed_file(input_file):
        file = open(input_file, 'r')
        lis = []
        for line in file.readlines():
            lis.append(line.lower())
        file.close()

        result_list = []
        punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        for message in lis:
            if not message:
                continue

            new_message = ""
            if message[-1] == '\n':
                new_message = message[:-2]
            else:
                new_message = message

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
    def trigram_probability_dict(in_file, k):
        input_list = Trigram.processed_file(in_file)
        word_count = defaultdict(int)
        for message in input_list:
            for word in message:
                word_count[word] += 1

        unkown_words = set()
        for word in word_count:
            if word_count[word] == 1:
                unkown_words.add(word)

        one_word_dict = defaultdict(int)
        two_word_dict = defaultdict(int)
        three_word_dict = defaultdict(int)

        for message in input_list:
            if not message:
                continue

            for i in range(len(message)):
                if message[i] in unkown_words:
                    message[i] = "unk"
                if i == 0:
                    two_word_dict["s/_" + message[i]] += 1
                elif i == 1:
                    two_word_dict[message[i-1]+"_"+message[i]] += 1
                    three_word_dict["s/_" + message[i-1]+"_"+message[i]] += 1
                else:
                    two_word_dict[message[i-1]+"_"+message[i]] += 1
                    three_word_dict[message[i-2] + "_" + message[i-1]+"_"+message[i]] += 1

            two_word_dict[message[-1] + "_/e"] += 1
            if len(message) == 1:
                three_word_dict["s/_" + message[-1] + "_/e"] += 1
            else:
                three_word_dict[message[-2] + "_" + message[-1] + "_/e"] += 1

        words = one_word_dict.keys()
        for word1 in words:
            for word2 in words:
                for word3 in words:
                    three_word_dict[word1 + "_" + word2 + "_" + word3] += k

        pro_dict = defaultdict(float)
        for sequence, num in three_word_dict.items():
            words = sequence.split('_')
            word1, word2, word3 = words[0], words[1], words[2]
            pro_dict[sequence] = num / (two_word_dict[word1 + "_" + word2] + k * len(words))
        return pro_dict

    def perplexity(pro_dict, validation_list):
        total = 0
        pp = 0
        for message in validation_list:
            if not message:
                continue

            if len(message) == 1:
                total += 1
                if pro_dict["s/_" + message[0] + "_/e"] != 0:
                    pp += -np.log(pro_dict["s/_" + message[0] + "_/e"])
                elif pro_dict["s/_" + "unk" + "_/e"] != 0:
                    pp += -np.log(pro_dict["s/_" + "unk" + "_/e"])
                else:
                    pp += 100
                continue

            for i in range(1, len(message)+1):
                total += 1
                if i == 1:
                    if pro_dict["s/_" + message[0] + message[1]] != 0:
                        pp += -np.log(pro_dict["s/_" + message[0] + "_" + message[1]])
                    elif pro_dict["s/_" + message[0] + "unk"] != 0:
                        pp += -np.log(pro_dict["s/_" + message[0] + "_unk"])
                    elif pro_dict["s/_" + "unk" + message[1]] != 0:
                        pp += -np.log(pro_dict["s/_" + "unk" + message[1]])
                    else:
                        pp += -np.log(pro_dict["s/_" + "unk" + "_unk"])
                elif i == len(message):
                    if pro_dict[message[-2] + "_" + message[-1] + "_/e"] != 0:
                        pp += -np.log(pro_dict[message[-2] + "_" + message[-1] + "_/e"])
                    elif pro_dict[message[-2] + "_unk" + "_/e"] != 0:
                        pp += -np.log(pro_dict[message[-2] + "_unk" + "_/e"])
                    elif pro_dict["unk" + "_" + message[-1] + "_/e"] != 0:
                        pp += -np.log(pro_dict["unk" + "_" + message[-1] + "_/e"])
                    else:
                        pp += -np.log(pro_dict["unk" + "_unk" + "_/e"])
                else:
                    if pro_dict[message[i] + "_" + message[i-1] + "_" + message[i-2]] != 0:
                        pp += -np.log(pro_dict[message[i] + "_" + message[i-1] + "_" + message[i-2]])
                    elif pro_dict[message[i] + "_" + message[i-1] + "_unk"] != 0:
                        pp += -np.log(pro_dict[message[i] + "_" + message[i-1] + "_unk"])
                    elif pro_dict[message[i] + "_" + "unk" + "_" + message[i-2]] != 0:
                        pp += -np.log(pro_dict[message[i] + "_" + "unk" + "_" + message[i-2]])
                    elif pro_dict["unk" + "_" + message[i-1] + "_" + message[i-2]] != 0:
                        pp += -np.log(pro_dict["unk" + "_" + message[i-1] + "_" + message[i-2]])
                    elif pro_dict[message[i] + "_unk" + "_unk"] != 0:
                        pp += -np.log(pro_dict[message[i] + "_unk" + "_unk"])
                    elif pro_dict["unk" + "_" + message[i-1] + "_" + "unk"] != 0:
                        pp += -np.log(pro_dict["unk" + "_" + message[i-1] + "_" + "unk"])
                    elif pro_dict["unk" + "_" + "unk" + "_" + message[i-2]] != 0:
                        pp += -np.log(pro_dict["unk" + "_" + "unk" + "_" + message[i-2]])
                    else:
                        pp += -np.log(pro_dict["unk" + "_unk" + "_unk"])

            return np.exp(pp / total)
            

    @staticmethod
    def perplexity_single_message(pro_dict, message):
        if not message:
            return 0.0

        pp = 0
        if len(message) == 1:
            if pro_dict["s/_" + message[0] + "_/e"] != 0:
                pp += -np.log(pro_dict["s/_" + message[0] + "_/e"])
            elif pro_dict["s/_" + "unk" + "_/e"] != 0:
                pp += -np.log(pro_dict["s/_" + "unk" + "_/e"])
            else:
                pp += 10
            return pp

        for i in range(1, len(message) + 1):
            if i == 1:
                if pro_dict["s/_" + message[0] + message[1]] != 0:
                    pp += -np.log(pro_dict["s/_" + message[0] + "_" + message[1]])
                elif pro_dict["s/_" + message[0] + "unk"] != 0:
                    pp += -np.log(pro_dict["s/_" + message[0] + "_unk"])
                else:
                    pp += -np.log(pro_dict["s/_" + "unk" + "_unk"])
            elif i == len(message):
                if pro_dict[message[-2] + "_" + message[-1] + "_/e"] != 0:
                    pp += -np.log(pro_dict[message[-2] + "_" + message[-1] + "_/e"])
                elif pro_dict[message[-2] + "_unk" + "_/e"] != 0:
                    pp += -np.log(pro_dict[message[-2] + "_unk" + "_/e"])
                else:
                    pp += -np.log(pro_dict["unk" + "_unk" + "_/e"])
            else:
                if pro_dict[message[i] + "_" + message[i - 1] + "_" + message[i - 2]] != 0:
                    pp += -np.log(pro_dict[message[i] + "_" + message[i - 1] + "_" + message[i - 2]])
                elif pro_dict[message[i] + "_" + message[i - 1] + "_unk"] != 0:
                    pp += -np.log(pro_dict[message[i] + "_" + message[i - 1] + "_unk"])
                elif pro_dict[message[i] + "_unk" + "_unk"] != 0:
                    pp += -np.log(pro_dict[message[i] + "_unk" + "_unk"])
                else:
                    pp += -np.log(pro_dict["unk" + "_unk" + "_unk"])
        return pp / (len(message))