from flask import Flask,request, render_template
import colorama
from colorama import Fore, Style
from allmodels.bigram.bigram import Bigram
from allmodels.bayes.bayes import Bayes
from allmodels.trigram.trigram import Trigram

app = Flask(__name__)
# load trained model
def get_test_result(model, vectorizer, input_sentence):
    test_data = [Bayes.processed_single_message(input_sentence)]
    test_doc = vectorizer.fit_transform(test_data)
    test_doc = test_doc.toarray()

    index = model.predict(test_doc[0].reshape(1, -1))
    reslist = [0, 1]
    return reslist[index[0]]

k = 0.01
ham_file = 'ham/train.txt'
spam_file = 'spam/train.txt'

# load bayes model
train_matrix, class_labels, vectorizer = Bayes.process_only_training_data(ham_file, spam_file)
bayes_model = Bayes.MyMultinomialNB_single_message(train_matrix, class_labels)

# load bigram model
ham_pro_dict_bigram = Bigram.bigram_probability_dict(ham_file, k)
spam_pro_dict_bigram = Bigram.bigram_probability_dict(spam_file, k)

# load trigram model
ham_pro_dict_trigram = Trigram.trigram_probability_dict(ham_file, k)
spam_pro_dict_trigram = Trigram.trigram_probability_dict(spam_file, k)

#define app routes
@app.route("/")
def index():
    return render_template("./index.html")

@app.route("/get")
#function for the bot response
def get_bot_response():
    user_input = request.args.get('msg')
    user_method = request.args.get('method');

    if user_method == 'bayers':
        result = get_test_result(bayes_model, vectorizer, user_input)
        if result == 0:
            return "This message is not spam."
        else:
            return "This message is spam."
    elif user_method == 'bigram':
        message = Bigram.processed_single_message(user_input)
        spam_pp = Bigram.perplexity(spam_pro_dict_bigram, message)
        ham_pp = Bigram.perplexity(ham_pro_dict_bigram, message)
        if spam_pp > ham_pp:
            return "This message is not spam."
        else:
            return "This message is spam."
    else:
        message = Trigram.processed_single_message(user_input)
        spam_pp = Trigram.perplexity(spam_pro_dict_trigram, message)
        ham_pp = Trigram.perplexity(ham_pro_dict_trigram, message)
        if spam_pp > ham_pp:
            return "This message is not spam."
        else:
            return "This message is spam."



if __name__ == "__main__":
    app.run()