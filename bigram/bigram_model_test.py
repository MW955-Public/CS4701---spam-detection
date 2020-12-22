import numpy as np
import csv
from bigram import Bigram

k = 0.01
ham_test_list = Bigram.processed_file("../ham/test.txt")
spam_test_list = Bigram.processed_file("../spam/test.txt")

ham_pro_dict = Bigram.bigram_probability_dict("../ham/train.txt", k)
spam_pro_dict = Bigram.bigram_probability_dict("../spam/train.txt", k)

ham_count_ham = ham_count_spam = 0
for message in ham_test_list:
    spam_pp = Bigram.perplexity_single_message(spam_pro_dict, message)
    ham_pp = Bigram.perplexity_single_message(ham_pro_dict, message)
    result = None
    if spam_pp > ham_pp:
        ham_count_ham += 1
    else:
        ham_count_spam += 1

spam_count_ham = spam_count_spam = 0
for message in spam_test_list:
    spam_pp = Bigram.perplexity_single_message(spam_pro_dict, message)
    ham_pp = Bigram.perplexity_single_message(ham_pro_dict, message)
    result = None
    if spam_pp > ham_pp:
        spam_count_ham += 1
    else:
        spam_count_spam += 1

ham_accuracy = float(ham_count_ham)/float(ham_count_ham+ham_count_spam)
spam_accuracy = float(spam_count_spam)/float(spam_count_ham+spam_count_spam)
recall= float(spam_count_spam)/float(spam_count_spam+spam_count_ham)
precision = float(spam_count_spam)/float(spam_count_spam+ham_count_spam)
f1_score = 2*precision*recall/(precision+recall)

print("ham accuracy: " + str(ham_accuracy))
print("spam accuracy: " + str(spam_accuracy))
print("precision: " + str(precision))
print("recall: " + str(recall))
print("F1 score: " + str(f1_score))