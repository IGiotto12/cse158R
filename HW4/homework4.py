# %%
import gzip
import math
import numpy as np
import random
import sklearn
import string
from collections import defaultdict
from collections import Counter
from nltk.stem.porter import *
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import dateutil
from scipy.sparse import lil_matrix # To build sparse feature matrices, if you like

# %%
answers = {}

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
### Question 1

# %%
dataset = []

f = gzip.open("steam_category.json.gz")
for l in f:
    d = eval(l)
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
Ntrain = 10000
Ntest = 10000

dataTrain = dataset[:Ntrain]
dataTest = dataset[Ntrain:Ntrain + Ntest]

# %%
sp = set(string.punctuation)
word_counts = Counter()

def preprocess_text(text):
    return ''.join([c for c in text.lower() if c not in sp])

for review in dataTrain:
    text = preprocess_text(review['text'])
    words = text.split()
    word_counts.update(words)

counts = word_counts.most_common(1000)


# %%
answers['Q1'] = [(count, word) for word, count in counts[:10]]
assertFloatList([x[0] for x in answers['Q1']], 10)

# %%
### Question 2

# %%
NW = 1000 # dictionary size

# %% [markdown]
# 

# %%
words = [word for word, _ in counts[:NW]]  # Top 1000 words from Q1

# %%
# Build X...
vectorizer = CountVectorizer(vocabulary=words)  # Use the top 1000 words as the vocabulary
X = vectorizer.fit_transform([review['text'].lower() for review in dataset])  # Convert reviews to feature matrix

# %%
y = [review['genreID'] for review in dataset]

# %%
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(Xtrain, ytrain)
predictions = mod.predict(Xtest)

# %%
correct = [predictions[i] == ytest[i] for i in range(len(ytest))]

# %%
answers['Q2'] = sum(correct) / len(correct)
assertFloat(answers['Q2'])

# %%
### Question 3

# %%
target_words = ["character", "game", "length", "a", "it"]

df = defaultdict(int)
for review in dataTrain:
    text = preprocess_text(review['text'])
    words = set(text.split())  # Unique words in the document
    for w in words:
        df[w] += 1

N = len(dataTrain)

# %%
# Calculate IDF for target words
idf = {}
for word in target_words:
    idf[word] = math.log10(N / (1 + df[word]))

first_review = preprocess_text(dataTrain[0]['text'])
first_review_words = first_review.split()

tf = {}
for word in target_words:
    tf[word] = sum(1 for w in first_review_words if w == word)

# Compute TF-IDF for the target words
tfidf = {}
for word in target_words:
    tfidf[word] = tf[word] * idf[word]

# %%
answers['Q3'] = [(idf[word], tfidf[word]) for word in target_words]

assertFloatList([x[0] for x in answers['Q3']], 5)
assertFloatList([x[1] for x in answers['Q3']], 5)

# %%
### Question 4

# %%
# Build X and y...

# %%
Xtrain = X[:Ntrain]
ytrain = y[:Ntrain]
Xtest = X[Ntrain:]
ytest = y[Ntrain:]

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(Xtrain, ytrain)

predictions = mod.predict(Xtest)
correct = [predictions[i] == ytest[i] for i in range(len(ytest))]

# %%
answers['Q4'] = sum(correct) / len(correct)
assertFloat(answers['Q4'])

# %%
### Question 5

# %%
def Cosine(x1, x2):
    dot_product = sum(x1.get(word, 0) * x2.get(word, 0) for word in x1.keys() | x2.keys())
    
    norm_x1 = math.sqrt(sum(value**2 for value in x1.values()))
    norm_x2 = math.sqrt(sum(value**2 for value in x2.values()))
    
    # edge cases
    if norm_x1 == 0 or norm_x2 == 0:
        return 0.0
    
    return dot_product / (norm_x1 * norm_x2)

# %%
# Compute IDF
def compute_idf(dataTrain):
    idf = defaultdict(float)
    doc_count = len(dataTrain)
    word_doc_count = Counter()

    for review in dataTrain:
        words = set(preprocess_text(review['text']).split())
        for word in words:
            word_doc_count[word] += 1

    for word, count in word_doc_count.items():
        idf[word] = math.log10(doc_count / count) if count > 0 else 0.0

    return idf

idf = compute_idf(dataTrain)

def compute_tfidf(text):
    words = preprocess_text(text).split()
    tf = defaultdict(int)
    for word in words:
        tf[word] += 1
    return {word: (tf[word] / len(words)) * idf.get(word, 0.0) for word in tf}

# Check TF-IDF and similarity computation
first_review_tfidf = compute_tfidf(dataTrain[0]['text'])
test_tfidfs = [(compute_tfidf(review['text']), review.get('reviewID', None)) for review in dataTest]

# %%
similarities = []
for test_tfidf, reviewID in test_tfidfs:
    similarity = Cosine(first_review_tfidf, test_tfidf)
    similarities.append((similarity, reviewID))

similarities.sort(reverse=True)

# %%
answers['Q5'] = similarities[0]
assertFloat(answers['Q5'][0])

# %%
### Question 6

# %%
# Define possible values for NW and C
dictionary_sizes = [500, 1000, 2000]
regularization_constants = [0.01, 0.1, 1, 10]

best_accuracy = 0
best_NW = None
best_C = None

# %%
# Iterate over different dict size
for NW in dictionary_sizes:
    words = [word for word, _ in counts[:NW]]
    vectorizer = TfidfVectorizer(vocabulary = words)
    X = vectorizer.fit_transform([review['text'].lower() for review in dataset])
    y = [review['genreID'] for review in dataset]
    
    Xtrain = X[:Ntrain]
    ytrain = y[:Ntrain]
    Xtest = X[Ntrain:]
    ytest = y[Ntrain:]

    # Iterate over different C values
    for C in regularization_constants:
        mod = linear_model.LogisticRegression(C=C)
        mod.fit(Xtrain, ytrain)

        predictions = mod.predict(Xtest)
        accuracy = accuracy_score(ytest, predictions)

        # Update best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_NW = NW
            best_C = C

# %%
answers['Q6'] = best_accuracy
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
import dateutil.parser

# %%
dataset = []

f = gzip.open("young_adult_20000.json.gz")
for l in f:
    d = eval(l)
    d['datetime'] = dateutil.parser.parse(d['date_added'])
    dataset.append(d)
    if len(dataset) >= 20000:
        break
        
f.close()

# %%
reviewLists = defaultdict(list)
for review in dataset:
    user = review['user_id']
    book = review['book_id']
    reviewLists[user].append((review['datetime'], book))

reviewLists = [sorted(reviews, key = lambda x: x[0]) for reviews in reviewLists.values()]
reviewLists = [[book for _, book in reviews] for reviews in reviewLists]  # Keep only book IDs

# %%
model5 = Word2Vec(reviewLists,
                  min_count=1, # Words/items with fewer instances are discarded
                  vector_size=5, # Model dimensionality
                  window=3, # Window size
                  sg=1) # Skip-gram model

# %%
first_book = reviewLists[0][0]
res = model5.wv.most_similar(first_book, topn = 5)

# %%
answers['Q7'] = res[:5]
assertFloatList([x[1] for x in answers['Q7']], 5)

# %%
f = open("answers_hw4.txt", 'w')
f.write(str(answers) + '\n')
f.close()


