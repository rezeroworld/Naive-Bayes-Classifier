import numpy as np
import pandas as pd
import re, unicodedata
from sklearn.model_selection import KFold
import sys

train_data, test_data = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')

def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return ''.join(new_words)

def split(text):
    # This function get rid of some symbols also
    text_ = (' ').join(text)
    text_ = (' ').join(text_.split('"'))
    text_ = (' ').join(text_.split("'"))
    text_ = (' ').join(text_.split('('))
    text_ = (' ').join(text_.split(')'))
    text_ = (' ').join(text_.split('['))
    text_ = (' ').join(text_.split(']'))
    text_ = (' ').join(text_.split(','))
    text_ = (' ').join(text_.split('.'))
    text_ = (' ').join(text_.split('-'))
    return text_.split(' ')

def remove_stop_words(words):
    words_ = ''.join(words).split(' ')
    # So I copied this stop words list from some source in the internet
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    new_words = []
    for word in words_:
        if word not in stop_words:
            new_words.append(word)
                 
    # Here we apply the split function multiple times to prevent nested cases like '('Viet')'
    for i in range(5):
        new_words = split(new_words)
        
    # This did not correct for all problems but it did help me to increase my accuracy just a little bit
    for word in list(new_words):
        if len(word) < 3 or word == 'non':
            new_words.remove(word)
        
    return (' ').join(new_words)

def process(df, t):
    df_copy = df.copy()
    df_copy[t] = df_copy[t].apply(lambda x : x.lower())
    df_copy[t] = df_copy[t].apply(lambda x : x.strip())
    df_copy[t] = df_copy[t].apply(lambda x : re.sub('\n', ' ', x))
    df_copy[t] = df_copy[t].apply(lambda x : re.sub('\[[^]]*\]', '', x))
    df_copy[t] = df_copy[t].apply(lambda x : re.sub('<.*?>', ' ', x))
    df_copy[t] = df_copy[t].apply(lambda x : remove_non_ascii(x))
    df_copy[t] = df_copy[t].apply(lambda x : remove_stop_words(x))
    
    return df_copy

train_data = process(train_data, 'Abstract')
test_data = process(test_data, 'Abstract')

X = train_data['Abstract'].to_numpy()
y = train_data['Category'].to_numpy()

# Uncomment this if you want to compute the predictions of the test set
#X_test = test_data['Abstract'].to_numpy()

class BernoulliVectorizer:
    def __init__(self):
        self.vocab = []
        self.vocab_counter = {}
        
    def build_vocab(self, data, vocab_size=5000):
        print('Building the vocab...')
        for document in data:
            for word in document.split(' '):
                if word in self.vocab:
                    self.vocab_counter[word] += 1
                else:
                    self.vocab.append(word)
                    self.vocab_counter[word] = 1
                            
        self.vocab_counter = dict(sorted(self.vocab_counter.items(), key=lambda x: x[1], reverse=True))
        self.vocab = list(self.vocab_counter.keys())
        print('vocab built originally with length', len(self.vocab), 'but considering only the ',vocab_size,' most repeated words')
        self.vocab = self.vocab[:20000]
    
    def transform(self, data):
        print('Transforming the data...')
        i = 0
        for document in data:
            bin_vect = np.zeros(len(self.vocab))
            for token in document.split(' '):
                for word_idx in range(len(self.vocab)):
                    if token == self.vocab[word_idx]:
                        bin_vect[word_idx] = 1
                        break
            data[i] = bin_vect
            i += 1
        print('transform complete')
        return data

    def fit_transform(self, data):
        self.build_vocab(data)
        return self.transform(data)
    
B = BernoulliVectorizer()
X = np.vstack(B.fit_transform(X))

# Uncomment this if you want to compute the predictions for the test set
#X_test = np.vstack(B.transform(X_test))

class BernoulliNB:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)

        # calculate all the priors
        counts = np.zeros_like(self.classes)
        for i in range(self.n_classes):
            counts[i] = np.sum((y == self.classes[i]).astype(int))
        self.classes_prob = (counts + self.alpha) / (len(y) + self.alpha * self.n_classes)
    
        # calculate the the number of occurence of each feature in each class
        self.params = np.zeros(shape=(self.n_classes, X.shape[1]))
        for i in range(self.n_classes):
            elts = X[y == self.classes[i],:]
            self.params[i] = (np.sum(elts ,axis=0) + self.alpha)/ (counts[i] + self.alpha * 2)

    def predict(self, X):
        neg_prob = np.log(1 - self.params)
        jll = np.dot(X, (np.log(self.params) - neg_prob).T)
        jll += np.log(self.classes_prob.astype(np.float64)) + neg_prob.sum(axis=1)
        return self.classes[np.argmax(jll, axis=1)]

# Create the model
B_model = BernoulliNB()

# Train the model
B_model.fit(X, y)

# See how the model performs on the train data
pred = B_model.predict(X)
print('accuracy on the training data is: ', np.mean(pred==y))

# Uncomment this and see lines 67 and 112 if you want the predictions for the test set 
'''
pred = B_model.predict(X_test)
pred = np.concatenate(pred)
pred = pd.DataFrame(pred)
pred.to_csv('pred_bernoulli.csv',index=True, index_label='Id', header=['Category'])
'''

# Just leave this, it was used when testing K-fold cross validation
'''
# Cross validation to have an estimator of the accuracy
B_model = BernoulliNB()
kf = KFold(n_splits=10)
s=0
i=0
for train_index, test_index in kf.split(X):
    B_model.fit(X[train_index], y[train_index])
    s += (B_model.predict(X[test_index]) == y[test_index]).mean()
    print((B_model.predict(X[test_index]) == y[test_index]).mean())
    i+=1
print('the average accuracy is: ',s/i)
'''