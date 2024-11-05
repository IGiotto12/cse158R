# %%
import json
import gzip
import math
import numpy as np
from collections import defaultdict
from sklearn import linear_model
import random
import statistics

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
# From https://cseweb.ucsd.edu/classes/fa24/cse258-b/files/steam.json.gz
z = gzip.open("steam.json.gz")

# %%
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

# %%
z.close()

# %%
### Question 1

# %%
def MSE(y, ypred):
    diffs = [(a-b)**2 for (a,b) in zip(y,ypred)]
    return sum(diffs) / len(diffs)

# %%
def feat1(d):
    return [1, len(d['text'])]

# %%
X = [feat1(d) for d in dataset]
y = [d['hours'] for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X, y)

y_pred = mod.predict(X)
mse1 = MSE(y, y_pred)

# %%
answers['Q1'] = [float(mod.coef_[1]), float(mse1)] # Remember to cast things to float rather than (e.g.) np.float64
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
dataTrain = dataset[:int(len(dataset)*0.8)]
dataTest = dataset[int(len(dataset)*0.8):]

# %%
X_train = [feat1(d) for d in dataTrain]
y_train = [d['hours'] for d in dataTrain]

X_test = [feat1(d) for d in dataTest]
y_test = [d['hours'] for d in dataTest]

mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X, y)

y_pred = mod.predict(X_test)
mse2 = MSE(y_test, y_pred)


# %%
under = 0
over = 0

for test, pred in zip(y_test, y_pred):
    if pred > test:
        over += 1
    if pred < test:
        under += 1

# %%
answers['Q2'] = [mse2, under, over]
assertFloatList(answers['Q2'], 3)

# %%
### Question 3

# %%
y2 = y[:]
y2.sort()
perc90 = y2[int(len(y2)*0.9)] # 90th percentile

X3a = []
y3a = []

for d in dataTrain:
    if d['hours'] <= perc90:
        X3a.append(feat1(d))
        y3a.append(d['hours'])

mod3a = linear_model.LinearRegression(fit_intercept=False)
mod3a.fit(X3a,y3a)
pred3a = mod3a.predict(X_test)

# %%
under3a = 0
over3a = 0

for test, pred in zip(y_test, pred3a):
    if pred > test:
        over3a += 1
    if pred < test:
        under3a += 1

# %%
# 3b

# %%
y3b = [d['hours_transformed'] for d in dataTrain]

mod3b = linear_model.LinearRegression(fit_intercept=False)
mod3b.fit(X_train,y3b)
pred3b = mod3b.predict(X_test)
pred3b_original = [2 ** pred - 1 for pred in pred3b]

# %%
under3b = 0
over3b = 0

for test, pred in zip(y_test, pred3b_original):
    if pred > test:
        over3b += 1
    if pred < test:
        under3b += 1

# %%
# 3c

# %%
theta0 = mod.coef_[0]

median_length = np.median([len(d['text']) for d in dataTrain])
median_hours = np.median([d['hours'] for d in dataTrain])

theta1 = (median_hours - theta0) / median_length

pred3c = [theta0 + theta1 * len(d['text']) for d in dataTest]

# %%
under3c = 0
over3c = 0

for test, pred in zip(y_test, pred3c):
    if pred > test:
        over3c += 1
    if pred < test:
        under3c += 1

# %%
answers['Q3'] = [under3a, over3a, under3b, over3b, under3c, over3c]
assertFloatList(answers['Q3'], 6)

# %%
### Question 4

# %%
y = [1 if d['hours'] > median_hours else 0 for d in dataTrain]
ytest = [1 if d['hours'] > median_hours else 0 for d in dataTest]


mod_log = linear_model.LogisticRegression(C=1)
mod_log.fit(X_train,y)
pred4 = mod_log.predict(X_test) # Binary vector of predictions

# %%
from sklearn.metrics import confusion_matrix

def Calc_BER(predictions, y):
    TN, FP, FN, TP = confusion_matrix(y, predictions).ravel()
    ber = 0.5 * ((FP / (TN + FP) if (TN + FP) > 0 else 0) + 
                 (FN / (TP + FN) if (TP + FN) > 0 else 0))
    return ber

TN, FP, FN, TP = confusion_matrix(ytest, pred4).ravel()
BER = Calc_BER(pred4, ytest)

# %%
answers['Q4'] = [TP, TN, FP, FN, BER]
assertFloatList(answers['Q4'], 5)

# %%
### Question 5

# %%
under5 = 0
over5 = 0

for test, pred in zip(ytest, pred4):
    if pred > test:
        over5 += 1
    if pred < test:
        under5 += 1

# %%
answers['Q5'] = [over5, under5]
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
# Part (a)
train_early = [d for d in dataTrain if int(d['date'][:4]) <= 2014]
test_early = [d for d in dataTest if int(d['date'][:4]) <= 2014]

median_hours_early = np.median([d['hours'] for d in train_early])
X2014train = [feat1(d) for d in train_early]
y2014train = [1 if d['hours'] > median_hours_early else 0 for d in train_early]

X2014test = [feat1(d) for d in test_early]
y2014test = [1 if d['hours'] > median_hours_early else 0 for d in test_early]

mod_log = linear_model.LogisticRegression(C=1)
mod_log.fit(X2014train, y2014train)
pred2014 = mod_log.predict(X2014test)

BER_A = Calc_BER(pred2014, y2014test)


# %%
# Part (b)
train_later = [d for d in dataTrain if int(d['date'][:4]) >= 2015]
test_later = [d for d in dataTest if int(d['date'][:4]) >= 2015]

median_hours_later = np.median([d['hours'] for d in train_later])
X2015train = [feat1(d) for d in train_later]
y2015train = [1 if d['hours'] > median_hours_later else 0 for d in train_later]

X2015test = [feat1(d) for d in test_later]
y2015test = [1 if d['hours'] > median_hours_later else 0 for d in test_later]

mod_log = linear_model.LogisticRegression(C=1)
mod_log.fit(X2015train, y2015train)
pred2015 = mod_log.predict(X2015test)

BER_B = Calc_BER(pred2015, y2015test)


# %%
# Part (c)
mod_log = linear_model.LogisticRegression(C=1)
mod_log.fit(X2014train, y2014train)
predc = mod_log.predict(X2015test)

BER_C = Calc_BER(predc, y2015test)


# %%
# Part (d)
mod_log = linear_model.LogisticRegression(C=1)
mod_log.fit(X2015train, y2015train)
predd= mod_log.predict(X2014test)

BER_D = Calc_BER(predd, y2014test)


# %%
answers['Q6'] = [BER_A, BER_B, BER_C, BER_D]
assertFloatList(answers['Q6'], 4)

# %%
### Question 7

# %%
# Useful data structures
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratings = {}

dataTrain = dataset[:int(len(dataset)*0.8)]
dataTest = dataset[int(len(dataset)*0.8):]

for d in dataTrain:
    user, item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

# Populate the dictionaries from training data
for d in dataTrain:
    user = d['userID']
    game = d['gameID']
    hours_transformed = d['hours_transformed']
    review_date = d['date']
    
    # Append review data to each user's list
    reviewsPerUser[user].append({
        'gameID': game, 
        'hours_transformed': hours_transformed,
        'date': review_date
    })
    
    # Append review data to each item's list
    reviewsPerItem[game].append({
        'userID': user, 
        'hours_transformed': hours_transformed,
        'date': review_date
    })


# %%
# Jaccard similarity
def Jaccard(s1, s2):
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union != 0 else 0

# find N most similar users
def mostSimilar(user, N):
    similarity = []
    target_items = itemsPerUser[user]

    for other_user, items in itemsPerUser.items():
        if other_user != user:
            sim = Jaccard(target_items, items)
            similarity.append((sim, other_user))

    most_similar_users = sorted(similarity, reverse=True)[:N]
    return most_similar_users

top10similar = mostSimilar(dataset[0]['userID'], 10)

first = top10similar[0][0]
tenth = top10similar[-1][0]

# %%
answers['Q7'] = [first, tenth]
assertFloatList(answers['Q7'], 2)


# %%
### Question 8

# %%
global_avg = np.mean([d['hours_transformed'] for d in dataTrain])

def user_to_user_predict(user, item):
    weighted_sum = 0
    sum_of_weights = 0
    
    if user not in reviewsPerUser:
        return global_avg
    if item not in reviewsPerItem:
        return global_avg

    # Iterate through other users who reviewed the same item
    for review in reviewsPerItem[item]:
        other_user = review['userID']
        if other_user != user:
            # Calculate Jaccard similarity between users
            sim = Jaccard(set(d['gameID'] for d in reviewsPerUser[user]),
                          set(d['gameID'] for d in reviewsPerUser[other_user]))
            
            weighted_sum += sim * review['hours_transformed']
            sum_of_weights += sim

    return weighted_sum / sum_of_weights if sum_of_weights > 0 else global_avg

def item_to_item_predict(user, item):
    weighted_sum = 0
    sum_of_weights = 0

    if item not in reviewsPerItem:
        return global_avg

    if user not in reviewsPerUser:
        return global_avg

    # Iterate through other items the user has reviewed
    for review in reviewsPerUser[user]:
        other_item = review['gameID']
        if other_item != item:
            # Calculate Jaccard similarity between items
            sim = Jaccard(set(d['userID'] for d in reviewsPerItem[item]),
                          set(d['userID'] for d in reviewsPerItem[other_item]))
            
            weighted_sum += sim * review['hours_transformed']
            sum_of_weights += sim

    return weighted_sum / sum_of_weights if sum_of_weights > 0 else global_avg

# %%
# Calculate MSE for each predictor on the test set
def compute_mse(predict_fn, test_data):
    errors = []
    for d in test_data:
        # Call predict_fn with only userID and gameID
        pred = predict_fn(d['userID'], d['gameID'])
        errors.append((d['hours_transformed'] - pred) ** 2)
    return np.mean(errors)

MSEU = compute_mse(user_to_user_predict, dataTest)
MSEI = compute_mse(item_to_item_predict, dataTest)


# %%
answers['Q8'] = [MSEU, MSEI]
assertFloatList(answers['Q8'], 2)

# %%
### Question 9

# %%
import math

def user_to_user_predict_with_time(user, item):
    weighted_sum = 0
    sum_of_weights = 0
    global_avg = np.mean([d['hours_transformed'] for d in dataTrain])

    if user not in reviewsPerUser:
        return global_avg
    if item not in reviewsPerItem:
        return global_avg

    target_review = next((r for r in reviewsPerUser[user] if r['gameID'] == item), None)
    if target_review is None:
        return global_avg
    target_year = target_review['date'].year

    for review in reviewsPerItem[item]:
        other_user = review['userID']
        if other_user != user:
            # Calculate Jaccard similarity between users
            sim = Jaccard(set(d['gameID'] for d in reviewsPerUser[user]),
                          set(d['gameID'] for d in reviewsPerUser[other_user]))

            other_year = review['date'].year
            time_decay = math.exp(-abs(target_year - other_year))

            weighted_sum += sim * time_decay * review['hours_transformed']
            sum_of_weights += sim * time_decay

    return weighted_sum / sum_of_weights if sum_of_weights > 0 else global_avg

# %%
MSE9 = compute_mse(user_to_user_predict_with_time, dataTest)

# %%
answers['Q9'] = [MSE9]
assertFloatList(answers['Q9'], 1)

# %%
if "float" in str(answers) or "int" in str(answers):
    print("it seems that some of your answers are not native python ints/floats;")
    print("the autograder will not be able to read your solution unless you convert them to ints/floats")

# %%
f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()


