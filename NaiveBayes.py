# Henry Dinh
# CS 6375.001
# Assignment 2 - Naive Bayes algorithm
# To test the program, read the README file for instructions

import os
import sys
import collections
import re

# Stores emails as dictionaries. email_file_name : Document (class defined below)
training_set = dict()
test_set = dict()

# ham = 0 for not spam, spam = 1 for is spam
classes = ["ham", "spam"]

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
                storage_dict.update({dir_entry_path : Document(text, bagOfWords(text), true_class)})

# counts frequency of each word in the text files and order of sequence doesn't matter
def bagOfWords(text):
    bagsofwords = [collections.Counter(re.findall(r'\w+', text))]
    return bagsofwords

# Training
# C is set of classes. 1 for spam and 0 for not spam
# D is the data set
def trainMultinomialNB():
    # Extract vocabulary done in Documents class, which is contained in data dictionaries
    # n is the number of documents
    n = len(training_set)
    # for each class in classes (i.e. ham and spam)
    for c in classes:
        # n_c is number of documents with true class c
        n_c = 0.0
        # text_c = concatenation of text of all docs in class (D, c)
        text_c = ""
        for i in training_set:
            if training_set[i].getTrueClass() == c:
                n_c += 1
                text_c += training_set[i].getText()
        prior = float(n_c) / float(n)
        # Count frequencies/tokens of each term in text_c in dictionary form (i.e. token : frequency)
        token_freqs = bagOfWords(text_c)
        # Calculate conditional probabilities for each token and sum using laplace smoothing and log-scale
        conditional_probability = 0.0
        for t in token_freqs:
            conditional_probability += ((token_freqs[t] + 1) / (len(text_c) + 1))


# Testing
def applyMultinomialNB():
    pass

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
    makeDataSet(training_set,training_spam_dir, classes[1])
    makeDataSet(training_set, training_ham_dir, classes[0])
    makeDataSet(test_set, test_spam_dir, classes[1])
    makeDataSet(test_set, test_ham_dir, classes[0])
    # Prints out the data set for testing purposes
    for i in test_set:
        print i + " : "
        print test_set[i].getText()
        print test_set[i].getWordFreqs()
        print test_set[i].getTrueClass()
        print

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])