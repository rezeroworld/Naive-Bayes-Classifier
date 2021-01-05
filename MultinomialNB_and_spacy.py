from spacy.lang.en import English
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import string

import pandas as pd

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


train_data = pd.read_csv('train.csv')

punctuations = string.punctuation

nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

parser = English()

def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    return mytokens

class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

def clean_text(text):
    return text.strip().lower()

#bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
bow_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

X_train, y_train = train_data['Abstract'], train_data['Category']

classifier = MLPClassifier(hidden_layer_sizes=(128,2))

pipe = Pipeline([("cleaner", predictors()),('vectorizer', bow_vector),('classifier', classifier)])

kf = KFold(n_splits=5)
s=0
i=0
for train_index, test_index in kf.split(X_train):
    pipe.fit(X_train[train_index], y_train[train_index])
    s += (pipe.predict(X_train[test_index]) == y_train[test_index]).mean()
    print((pipe.predict(X_train[test_index]) == y_train[test_index]).mean())
    i+=1
print('the average accuracy is: ',s/i)