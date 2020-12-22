###cross validation splitting datasets
a=[i for i in range(int(4800*0.8))]
b=[i for i in range(int(4800*0.8),int(4800))]
c=[i for i in range(int(700*0.8))]
d=[i for i in range(int(700*0.8),int(700*1.0))]

import csv

ham_count = 0
spam_count = 0
ham_messages = []
spam_messages = []
with open('./spam.csv', newline='') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    line_count = 0
    for row in raw_data:
        if line_count != 0:
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])
        line_count += 1

file = open("ham/train_1.txt", "w")
for i in a:
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("./ham/validation_1.txt", "w")
for i in b:
    file.write(ham_messages[i])
    file.write("\n")
file.close()


file = open("spam/train_1.txt", "w")
for i in c:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("./spam/validation_1.txt", "w")
for i in d:
    file.write(spam_messages[i])
    file.write("\n")
file.close()


###cross validation
a=[i for i in range(int(4800*0.6))]+[i for i in range(int(4800*0.8),int(4800))]
b=[i for i in range(int(4800*0.6),int(4800*0.8))]
c=[i for i in range(int(700*0.6))]+[i for i in range(int(700*0.8),int(700))]
d=[i for i in range(int(700*0.6),int(700*0.8))]

ham_count = 0
spam_count = 0
ham_messages = []
spam_messages = []
with open('./spam.csv', newline='') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    line_count = 0
    for row in raw_data:
        if line_count != 0:
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])
        line_count += 1

file = open("ham/train_2.txt", "w")
for i in a:
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("./ham/validation_2.txt", "w")
for i in b:
    file.write(ham_messages[i])
    file.write("\n")
file.close()


file = open("spam/train_2.txt", "w")
for i in c:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("./spam/validation_2.txt", "w")
for i in d:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

###cross validation
a=[i for i in range(int(4800*0.4))]+[i for i in range(int(4800*0.6),int(4800))]
b=[i for i in range(int(4800*0.4),int(4800*0.6))]
c=[i for i in range(int(700*0.4))]+[i for i in range(int(700*0.6),int(700))]
d=[i for i in range(int(700*0.4),int(700*0.6))]

ham_count = 0
spam_count = 0
ham_messages = []
spam_messages = []
with open('./spam.csv', newline='') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    line_count = 0
    for row in raw_data:
        if line_count != 0:
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])
        line_count += 1

file = open("ham/train_3.txt", "w")
for i in a:
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("./ham/validation_3.txt", "w")
for i in b:
    file.write(ham_messages[i])
    file.write("\n")
file.close()


file = open("spam/train_3.txt", "w")
for i in c:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("./spam/validation_3.txt", "w")
for i in d:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

###cross validation
a=[i for i in range(int(4800*0.2))]+[i for i in range(int(4800*0.4),int(4800))]
b=[i for i in range(int(4800*0.2),int(4800*0.4))]
c=[i for i in range(int(700*0.2))]+[i for i in range(int(700*0.4),int(700))]
d=[i for i in range(int(700*0.2),int(700*0.4))]

ham_count = 0
spam_count = 0
ham_messages = []
spam_messages = []
with open('./spam.csv', newline='') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    line_count = 0
    for row in raw_data:
        if line_count != 0:
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])
        line_count += 1

file = open("ham/train_4.txt", "w")
for i in a:
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("./ham/validation_4.txt", "w")
for i in b:
    file.write(ham_messages[i])
    file.write("\n")
file.close()


file = open("spam/train_4.txt", "w")
for i in c:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("./spam/validation_4.txt", "w")
for i in d:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

###cross validation
a=[i for i in range(int(4800*0.2),int(4800))]
b=[i for i in range(int(4800*0.2))]
c=[i for i in range(int(700*0.2),int(700))]
d=[i for i in range(int(700*0.2))]

ham_count = 0
spam_count = 0
ham_messages = []
spam_messages = []
with open('./spam.csv', newline='') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    line_count = 0
    for row in raw_data:
        if line_count != 0:
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])
        line_count += 1

file = open("ham/train_5.txt", "w")
for i in a:
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("./ham/validation_5.txt", "w")
for i in b:
    file.write(ham_messages[i])
    file.write("\n")
file.close()


file = open("spam/train_5.txt", "w")
for i in c:
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("./spam/validation_5.txt", "w")
for i in d:
    file.write(spam_messages[i])
    file.write("\n")
file.close()


###cross validation performance

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

# now do the prediction, when you run the code change the file path to train_1, train_2, train_3, train_4, train_5, test_1....
ham_file = './ham/train_1.txt'
spam_file = './spam/train_1.txt'
test_file = './ham/test_1.txt'
train_matrix, class_labels, test_doc = Bayes.process_include_test_data(ham_file, spam_file, test_file)
test_file_2 = './spam/test_1.txt'
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
