# %%
import json
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy
import random
import gzip
import math

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x): # Checks that an answer is a float
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(json.loads(l))

# %%
len(dataset)

# %%
answers = {} # Put your answers to each question in this dictionary

# %%
dataset[0]

# %%
### Question 1

# %%
def feature(datum):
    # your implementation
    return[1, datum['review_text'].count('!')]

# %%
X = [feature(d) for d in dataset]
Y = [d['rating'] for d in dataset]

# %%
theta, residual, rank, s = numpy.linalg.lstsq(X, Y)

y_pred = X @ theta
theta0, theta1, mse = theta[0], theta[1], numpy.mean((Y - y_pred)**2)

# %%
answers['Q1'] = [theta0, theta1, mse]
assertFloatList(answers['Q1'], 3) # Check the format of your answer (three floats)

# %%
### Question 2

# %%
def feature(datum):
    return [1, datum['review_text'].count('!'), len(datum['review_text'])]

# %%
X = [feature(d) for d in dataset]
Y = [d['rating'] for d in dataset]

# %%
theta, residual, rank, s = numpy.linalg.lstsq(X, Y)

y_pred = X @ theta
theta0, theta1, theta2, mse = theta[0], theta[1], theta[2], numpy.mean((Y - y_pred)**2)

# %%
answers['Q2'] = [theta0, theta1, theta2, mse]
assertFloatList(answers['Q2'], 4)


# %%
### Question 3

# %%
def feature(datum, deg):
    # feature for a specific polynomial degree
    features = [1]
    for d in range(1, deg+1):
        features.append(datum['review_text'].count('!') ** d)
    return features

# %%
mses = []
for i in range(1, 6):
    X = [feature(d, i) for d in dataset]
    Y = [d['rating'] for d in dataset]
    theta = numpy.linalg.lstsq(X, Y)[0]
    y_pred = X @ theta
    mses.append(numpy.mean((Y - y_pred)**2))

# %%
answers['Q3'] = mses
assertFloatList(answers['Q3'], 5)

# %%
### Question 4

# %%
def feature(datum, deg):
    # feature for a specific polynomial degree
    features = [1]
    for d in range(1, deg+1):
        features.append(datum['review_text'].count('!') ** d)
    return features

# %%
mses = []
for i in range(1, 6):
    X = [feature(d, i) for d in dataset]
    X_train = X[:len(X)//2] # first half for training
    X_test = X[len(X)//2:] # second half for test
    Y = [d['rating'] for d in dataset]
    Y_train = Y[:len(Y)//2] # first half for training
    Y_test = Y[len(Y)//2:] # second half for tes
    
    theta = numpy.linalg.lstsq(X_train, Y_train)[0]
    y_pred = X_test @ theta
    mses.append(numpy.mean((Y_test - y_pred)**2))

# %%
answers['Q4'] = mses
assertFloatList(answers['Q4'], 5)
mses

# %%
### Question 5

# %%
X = [1]*10000
Y = [d['rating'] for d in dataset]
theta = numpy.median(Y)

# %%
mae = numpy.mean(abs(Y - theta))

# %%
answers['Q5'] = mae
assertFloat(answers['Q5'])

# %%
### Question 6

# %%
f = open("beer_50000.json")
dataset = []
for l in f:
    if 'user/gender' in l:
        dataset.append(eval(l))

# %%
len(dataset)
dataset[0]

# %%
def feature(datum):
    return [1, datum['review/text'].count('!')]

# %%
X = [feature(d) for d in dataset if d['user/gender'] in ['Male', 'Female']]
y = [1 if d['user/gender'] == 'Female' else 0 for d in dataset if d['user/gender'] in ['Male', 'Female']]


# %%
model = linear_model.LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)


# %%
TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()

# Calculate BER
total_male = TN + FP
total_female = TP + FN
BER = 0.5 * (FP / total_male + FN / total_female)

# %%
answers['Q6'] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q6'], 5)

# %%
### Question 7

# %%
balanced_model = linear_model.LogisticRegression(class_weight='balanced')
balanced_model.fit(X, y)

y_pred = balanced_model.predict(X)


# %%
TN, FP, FN, TP = confusion_matrix(y, y_pred).ravel()
BER = 0.5 * (FP / (TN + FP) + FN / (TP + FN))

# %%
answers["Q7"] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q7'], 5)

# %%
### Question 8

# %%
def precision_at_k(y_true, y_scores, k):
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    top_k = sorted_indices[:k]
    # Count true positives
    true_positives = sum([y_true[i] for i in top_k])
    return true_positives / k

y_scores = balanced_model.decision_function(X)

# Calculate Precision@K for K ∈ [1, 10, 100, 1000, 10000]
k_values = [1, 10, 100, 1000, 10000]
precisionList = []
for k in k_values:
    if k <= len(y):
        precisionList.append(precision_at_k(y, y_scores, k))
    else:
        precisionList.append(0.0)

# %%
answers['Q8'] = precisionList
assertFloatList(answers['Q8'], 5) #List of five floats

# %%
f = open("answers_hw1.txt", 'w') # Write your answers to a file
f.write(str(answers) + '\n')
f.close()


