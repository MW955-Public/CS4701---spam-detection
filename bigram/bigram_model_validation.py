import numpy as np
import csv
from bigram import Bigram

ham_validation_list = Bigram.processed_file("../ham/validation.txt")
spam_validation_list = Bigram.processed_file("../spam/validation.txt")

with open('eval_k_0.1_1.0.csv', mode='w') as report_file:
    report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    report_writer.writerow(['k', 'ham accuracy', 'spam accuracy', 'precision', 'recall', 'F1_score'])

    for k in np.linspace(0.1, 1.0, 10):
        ham_pro_dict = Bigram.bigram_probability_dict("../ham/train.txt", k)
        spam_pro_dict = Bigram.bigram_probability_dict("../spam/train.txt", k)

        ham_count_ham = ham_count_spam = 0
        for message in ham_validation_list:
            spam_pp = Bigram.perplexity_single_message(spam_pro_dict, message)
            ham_pp = Bigram.perplexity_single_message(ham_pro_dict, message)
            result = None
            if spam_pp > ham_pp:
                ham_count_ham += 1
            else:
                ham_count_spam += 1

        spam_count_ham = spam_count_spam = 0
        for message in spam_validation_list:
            spam_pp = Bigram.perplexity_single_message(spam_pro_dict, message)
            ham_pp = Bigram.perplexity_single_message(ham_pro_dict, message)
            result = None
            if spam_pp > ham_pp:
                spam_count_ham += 1
            else:
                spam_count_spam += 1

        ham_accuracy = float(ham_count_ham)/float(ham_count_ham+ham_count_spam)
        spam_accuracy = float(spam_count_spam)/float(spam_count_ham+spam_count_spam)
        recall = float(spam_count_spam)/float(spam_count_spam+spam_count_ham)
        precision = float(spam_count_spam)/float(spam_count_spam+ham_count_spam)
        f1_score = 2*precision*recall/(precision+recall)

        report_writer.writerow([str(k), str(ham_accuracy), str(spam_accuracy), str(precision), str(recall),
                                str(f1_score)])

with open('eval_k_0.01_0.1.csv', mode='w') as report_file:
    report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    report_writer.writerow(['k', 'ham accuracy', 'spam accuracy', 'precision', 'recall', 'F1_score'])

    for k in np.linspace(0.01, 0.1, 10):
        ham_pro_dict = Bigram.bigram_probability_dict("../ham/train.txt", k)
        spam_pro_dict = Bigram.bigram_probability_dict("../spam/train.txt", k)

        ham_count_ham = ham_count_spam = 0
        for message in ham_validation_list:
            spam_pp = Bigram.perplexity_single_message(spam_pro_dict, message)
            ham_pp = Bigram.perplexity_single_message(ham_pro_dict, message)
            result = None
            if spam_pp > ham_pp:
                ham_count_ham += 1
            else:
                ham_count_spam += 1

        spam_count_ham = spam_count_spam = 0
        for message in spam_validation_list:
            spam_pp = Bigram.perplexity_single_message(spam_pro_dict, message)
            ham_pp = Bigram.perplexity_single_message(ham_pro_dict, message)
            result = None
            if spam_pp > ham_pp:
                spam_count_ham += 1
            else:
                spam_count_spam += 1

        ham_accuracy = float(ham_count_ham)/float(ham_count_ham+ham_count_spam)
        spam_accuracy = float(spam_count_spam)/float(spam_count_ham+spam_count_spam)
        precision = float(spam_count_spam)/float(spam_count_spam+spam_count_ham)
        recall = float(spam_count_spam)/float(spam_count_spam+ham_count_spam)
        f1_score = 2*precision*recall/(precision+recall)

        report_writer.writerow([str(k), str(ham_accuracy), str(spam_accuracy), str(precision), str(recall),
                                str(f1_score)])