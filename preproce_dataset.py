# Since there are 5572 records of messages in the raw data file, 4826 of them are useful, and only 746 of them are spam,
# the training datasets are not very balance. For the useful messages, we make use of the first 4800 records, and for
# the spam messages, we use the first 700 records. Of the two different types, we use 70% of messages as training data,
# 20% of them as validation data, and 10% of them as test data.

import csv

ham_count = 0
spam_count = 0
ham_messages = []
spam_messages = []
with open('spam.csv', newline='') as raw_data_file:
    raw_data = csv.reader(raw_data_file, delimiter=',')
    line_count = 0
    for row in raw_data:
        if line_count != 0:
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])
        line_count += 1

file = open("ham/train.txt", "w")
for i in range(int(4800*0.7)):
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("ham/validation.txt", "w")
for i in range(int(4800*0.7), int(4800*0.9)):
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("ham/test.txt", "w")
for i in range(int(4800*0.9), 4800):
    file.write(ham_messages[i])
    file.write("\n")
file.close()

file = open("spam/train.txt", "w")
for i in range(int(700*0.7)):
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("spam/validation.txt", "w")
for i in range(int(700*0.7), int(700*0.9)):
    file.write(spam_messages[i])
    file.write("\n")
file.close()

file = open("spam/test.txt", "w")
for i in range(int(700*0.9), 700):
    file.write(spam_messages[i])
    file.write("\n")
file.close()