from bayes import Bayes

# calculate validation accuracy
def accuracy(type, result):
    if type == 'ham':
        num = len(result) - sum(result)
        return float(num / len(result))
    elif type == 'spam':
        num = sum(result)
        return float(num / len(result))
    return 0

# now do the prediction, when you run the code change the file path
ham_file = '../ham/train.txt'
spam_file = '../spam/train.txt'
test_file = '../ham/test.txt'
test_file_2 = '../spam/test.txt'
train_matrix, class_labels, test_doc = Bayes.process_include_test_data(ham_file, spam_file, test_file)
train_matrix_2, class_labels_2, test_doc_2 = Bayes.process_include_test_data(ham_file, spam_file, test_file_2)
result = Bayes.MyMultinomialNB(train_matrix, class_labels, test_doc)
result2 = Bayes.MyMultinomialNB(train_matrix_2, class_labels_2, test_doc_2)
print('accuracy of ham: ' + str(accuracy('ham', result)))
print('accuracy of spam: ' + str(accuracy('spam', result2)))

precision=len(test_doc_2)*accuracy('spam', result2)/(len(test_doc_2)*accuracy('spam', result2)+len(test_doc)*(1-accuracy('ham', result)))
recall = accuracy('spam', result2)
F_1=2*precision*recall/(precision+recall)
print('precision '+str(precision))
print('recall ' +str(recall))
print('F_1 '+str(F_1))
