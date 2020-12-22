from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import unicodedata
import sys


class Bayes:

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

            result = new_message.translate(punctuation)
            result_list.append(result)
        return result_list

    @staticmethod
    # preprocess a single message
    def processed_single_message(input_message):
        if not input_message:
            return []

        new_message = ""
        if input_message[-1] == '\n':
            new_message = input_message[:-2]
        else:
            new_message = input_message

        punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
        result = new_message.translate(punctuation)
        return result

    @staticmethod
    def MyMultinomialNB(train_matrix='', class_labels='', test_doc=''):
        X = np.array(train_matrix)
        Y = np.array(class_labels)
        clf = MultinomialNB()
        clf.fit(X, Y)
        result = []
        for x in test_doc:
            index = clf.predict(x.reshape(1, -1))
            reslist = [0, 1]
            result.append(reslist[index[0]])
        return result

    def MyMultinomialNB_single_message(train_matrix='', class_labels=''):
        X = np.array(train_matrix)
        Y = np.array(class_labels)
        clf = MultinomialNB()
        clf.fit(X, Y)
        return clf

    # processing all the data, including the test data
    @staticmethod
    def process_include_test_data(ham_file, spam_file, test_file):
        train_matrix, class_labels, my_vectorizer = Bayes.process_only_training_data(ham_file, spam_file)
        test_data = Bayes.processed_file(test_file)
        test_doc = my_vectorizer.fit_transform(test_data).toarray()
        return (train_matrix, class_labels, test_doc)

    # processing the training data only
    def process_only_training_data(ham_file, spam_file):
        vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
        ham_processed = Bayes.processed_file(ham_file)
        spam_processed = Bayes.processed_file(spam_file)
        train = ham_processed + spam_processed
        train_matrix = vectorizer.fit_transform(train).toarray()
        feature_name = vectorizer.get_feature_names()
        my_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1, vocabulary=feature_name)
        class_labels = []
        for i in range(0, len(ham_processed)):
            class_labels.append(0)
        for i in range(0, len(spam_processed)):
            class_labels.append(1)
        return (train_matrix, class_labels, my_vectorizer)