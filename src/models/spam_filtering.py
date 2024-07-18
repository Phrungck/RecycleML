import re
import numpy as np

from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self):
        self.spam_word_count = defaultdict(int)
        self.ham_word_count = defaultdict(int)
        self.spam_total_words = 0
        self.ham_total_words = 0
        self.spam_priori = 0
        self.ham_priori = 0
    
    def preprocess(self, text):
        # our dataframe is already in lowercase, but for future purposes we add this
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
    
    def train(self, X, y):
        # filter Email data (X) with indices where Class (y) is equal to 1 or spam
        spam_messages = X[y == 1]
        # filter Email data (X) with indices where Class (y) is equal to 0 or ham
        ham_messages = X[y == 0]
        
        self.spam_priori = len(spam_messages) / len(X)
        self.ham_priori = len(ham_messages) / len(X)
        
        for message in spam_messages:
            words = self.preprocess(message)
            for word in words:
                # update dictionary with word and add value of 1 in value
                self.spam_word_count[word] += 1
                self.spam_total_words += 1
        
        for message in ham_messages:
            words = self.preprocess(message)
            for word in words:
                # update dictionary with word and add value of 1 in value
                self.ham_word_count[word] += 1
                self.ham_total_words += 1
    
    def predict(self, X, alpha=0):
        predictions = []
        for message in X:
            words = self.preprocess(message)
            spam_prob = np.log(self.spam_priori)
            ham_prob = np.log(self.ham_priori)
            
            for word in words:
                spam_prob += np.log((self.spam_word_count[word] + alpha) / (self.spam_total_words + len(self.spam_word_count) * alpha))
                ham_prob += np.log((self.ham_word_count[word] + alpha) / (self.ham_total_words + len(self.ham_word_count) * alpha))
            
            if spam_prob > ham_prob:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return np.array(predictions)