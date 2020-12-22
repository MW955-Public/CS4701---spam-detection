import colorama
from colorama import Fore, Style
from allmodels.bigram.bigram import Bigram
from allmodels.bayes.bayes import Bayes
from allmodels.trigram.trigram import Trigram


def get_test_result(model, vectorizer, input_sentence):
    test_data = [Bayes.processed_single_message(input_sentence)]
    test_doc = vectorizer.fit_transform(test_data)
    test_doc = test_doc.toarray()

    index = model.predict(test_doc[0].reshape(1, -1))
    reslist = [0, 1]
    return reslist[index[0]]


def chat():
    # load trained model
    k = 0.01
    ham_file = '/Users/luyingliu/PycharmProjects/PracInAI/allmodels/ham/train.txt'
    spam_file = '/Users/luyingliu/PycharmProjects/PracInAI/allmodels/spam/train.txt'

    # load bayes model
    train_matrix, class_labels, vectorizer = Bayes.process_only_training_data(ham_file, spam_file)
    bayes_model = Bayes.MyMultinomialNB_single_message(train_matrix, class_labels)

    # load bigram model
    ham_pro_dict_bigram = Bigram.bigram_probability_dict(ham_file, k)
    spam_pro_dict_bigram = Bigram.bigram_probability_dict(spam_file, k)

    # load trigram model
    ham_pro_dict_trigram = Trigram.trigram_probability_dict(ham_file, k)
    spam_pro_dict_trigram = Trigram.trigram_probability_dict(spam_file, k)

    model = ""
    while True:
        print(Fore.LIGHTGREEN_EX + "This system is to detect whether the message is spam. Please choose from following "
                                   "models: bigram, trigram, bayes." + Style.RESET_ALL)
        print(Fore.LIGHTRED_EX + "User: " + Style.RESET_ALL, end="")
        user_input = input().lower()
        if user_input in ["bigram", "trigram", "bayes"]:
            model = user_input
            break

    print(Fore.LIGHTGREEN_EX + "Please input your message (type 'quit' to stop)." + Style.RESET_ALL)

    while True:
        print(Fore.LIGHTRED_EX + "User: " + Style.RESET_ALL, end="")
        user_input = input().lower()
        if user_input == "quit":
            break

        print(Fore.LIGHTBLUE_EX + "ChatBot:" + Style.RESET_ALL, end="")
        result = ""
        if model == "bigram":
            message = Bigram.processed_single_message(user_input)
            spam_pp = Bigram.perplexity(spam_pro_dict_bigram, message)
            ham_pp = Bigram.perplexity(ham_pro_dict_bigram, message)
            if spam_pp > ham_pp:
                print("This message is not spam.")
            else:
                print("This message is spam.")

        elif model == "trigram":
            message = Trigram.processed_single_message(user_input)
            spam_pp = Trigram.perplexity(spam_pro_dict_trigram, message)
            ham_pp = Trigram.perplexity(ham_pro_dict_trigram, message)
            if spam_pp > ham_pp:
                print("This message is not spam.")
            else:
                print("This message is spam.")
        else:
            result = get_test_result(bayes_model, vectorizer, user_input)
            if result == 0:
                print("This message is not spam.")
            else:
                print("This message is spam.")


colorama.init()
chat()
