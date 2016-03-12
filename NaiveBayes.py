# Henry Dinh
# CS 6375.001
# Assignment 2 - Naive Bayes algorithm
# To test the program, read the README file for instructions

import os
import sys
import collections
import re
import math
import copy

# Stores emails as dictionaries. email_file_name : Document (class defined below)
training_set = dict()
test_set = dict()

# Filtered sets without stop words
filtered_training_set = dict()
filtered_test_set = dict()

# list of Stop words
stop_words = []

# ham = 0 for not spam, spam = 1 for is spam
classes = ["ham", "spam"]

# Conditional probability from the training data
conditional_probability = dict()
filtered_conditional_probability = dict()
# Prior for the classifications using the training data
prior = dict()
filtered_prior = dict()


# Read all text files in the given directory and construct the data set, D
# the directory path should just be like "train/ham" for example
# storage is the dictionary to store the email in
# True class is the true classification of the email (spam or ham)
def makeDataSet(storage_dict, directory, true_class):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                # stores dictionary of dictionary of dictionary as explained above in the initialization
                text = text_file.read()
                storage_dict.update({dir_entry_path: Document(text, bagOfWords(text), true_class)})


# counts frequency of each word in the text files and order of sequence doesn't matter
def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


# Extracts the vocabulary of all the text in a data set
def extractVocab(data_set):
    all_text = ""
    v = []
    for x in data_set:
        all_text += data_set[x].getText()
    for y in bagOfWords(all_text):
        v.append(y)
    return v


# Set the stop words
def setStopWords():
    stops = []
    with open('stop_words.txt', 'r') as txt:
        stops = (txt.read().splitlines())
    return stops


# Remove stop words from data set and store in dictionary
def removeStopWords(stops, data_set):
    filtered_data_set = copy.deepcopy(data_set)
    for i in stops:
        for j in filtered_data_set:
            if i in filtered_data_set[j].getWordFreqs():
                del filtered_data_set[j].getWordFreqs()[i]
    return filtered_data_set


# Training
def trainMultinomialNB(training, priors, cond):
    # v is the vocabulary of the training set
    v = extractVocab(training)
    # n is the number of documents
    n = len(training)
    # for each class in classes (i.e. ham and spam)
    for c in classes:
        # n_c is number of documents with true class c
        n_c = 0.0
        # text_c = concatenation of text of all docs in class (D, c)
        text_c = ""
        for i in training:
            if training[i].getTrueClass() == c:
                n_c += 1
                text_c += training[i].getText()
        priors[c] = float(n_c) / float(n)
        # Count frequencies/tokens of each term in text_c in dictionary form (i.e. token : frequency)
        token_freqs = bagOfWords(text_c)
        # Calculate conditional probabilities for each token and sum using laplace smoothing and log-scale
        for t in v:
            if t in token_freqs:
                cond.update({t + "_" + c: (float((token_freqs[t] + 1.0)) / float((len(text_c) + len(token_freqs))))})
            else:
                cond.update({t + "_" + c: (float(1.0) / float((len(text_c) + len(token_freqs))))})


# Testing. Data instance is a Document
# Returns classification guess
def applyMultinomialNB(data_instance, priors, cond):
    score = {}
    for c in classes:
        score[c] = math.log10(float(priors[c]))
        for t in data_instance.getWordFreqs():
            if (t + "_" + c) in cond:
                score[c] += float(math.log10(cond[t + "_" + c]))
    if score["spam"] > score["ham"]:
        return "spam"
    else:
        return "ham"




# Document class to store email instances easier
class Document:
    text = ""
    word_freqs = {}

    # spam or ham
    true_class = ""
    learned_class = ""

    # Constructor
    def __init__(self, text, counter, true_class):
        self.text = text
        self.word_freqs = counter
        self.true_class = true_class

    def getText(self):
        return self.text

    def getWordFreqs(self):
        return self.word_freqs

    def getTrueClass(self):
        return self.true_class

    def getLearnedClass(self):
        return self.learned_class

    def setLearnedClass(self, guess):
        self.learned_class = guess


# takes directories holding the data text files as paramters. "train/ham" for example
def main(training_spam_dir, training_ham_dir, test_spam_dir, test_ham_dir):
    # Set up data sets. Dictionaries containing the text, word frequencies, and true/learned classifications
    makeDataSet(training_set, training_spam_dir, classes[1])
    makeDataSet(training_set, training_ham_dir, classes[0])
    makeDataSet(test_set, test_spam_dir, classes[1])
    makeDataSet(test_set, test_ham_dir, classes[0])

    # Set the stop words list
    stop_words = setStopWords()

    # Set up data sets without stop words
    filtered_training_set = removeStopWords(stop_words, training_set)
    filtered_test_set = removeStopWords(stop_words, test_set)

    # Train using the training data
    trainMultinomialNB(training_set, prior, conditional_probability)
    trainMultinomialNB(filtered_training_set, filtered_prior, filtered_conditional_probability)

    # Test using the testing data - unfiltered
    correct_guesses = 0
    for i in test_set:
        test_set[i].setLearnedClass(applyMultinomialNB(test_set[i], prior, conditional_probability))
        if test_set[i].getLearnedClass() == test_set[i].getTrueClass():
            correct_guesses += 1

    # Test using the testing data - filtered
    correct_guesses_filtered = 0
    for i in filtered_test_set:
        filtered_test_set[i].setLearnedClass(applyMultinomialNB(filtered_test_set[i], filtered_prior,
                                                                filtered_conditional_probability))
        if filtered_test_set[i].getLearnedClass() == filtered_test_set[i].getTrueClass():
            correct_guesses_filtered += 1

    print "Correct guesses before filtering stop words:%d/%s" % (correct_guesses, len(test_set))
    print "Accuracy before filtering stop words:\t\t%.4f%%" % (100.0 * float(correct_guesses) / float(len(test_set)))
    print
    print "Correct guesses after filtering stop words:\t%d/%s" % (correct_guesses_filtered, len(filtered_test_set))
    print "Accuracy after filtering stop words:\t\t%.4f%%" % (100.0 * float(correct_guesses_filtered) / float(len(filtered_test_set)))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])