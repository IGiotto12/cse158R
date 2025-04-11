# %%
import numpy
import urllib
import scipy.optimize
import random
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import gzip
from collections import defaultdict

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
f = open("5year.arff", 'r')

# %%
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
answers = {} # Your answers

# %%
def accuracy(predictions, y):
    correct = sum (p == actual for p, actual in zip(predictions, y))
    return correct / len(y)

# %%
def BER(predictions, y):
    TN, FP, FN, TP = confusion_matrix(y, predictions).ravel()
    return 0.5 * (FP/(TN+FP) + FN/(TP+FN))

# %%
### Question 1

# %%
mod = linear_model.LogisticRegression(C=1)
mod.fit(X,y)

pred = mod.predict(X)

# %%
acc1 = accuracy(pred,y)
ber1 = BER(pred,y)

# %%
answers['Q1'] = [acc1, ber1] # Accuracy and balanced error rate

# %%
assertFloatList(answers['Q1'], 2)

# %%
### Question 2

# %%
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(X,y)

pred = mod.predict(X)

# %%
acc2 = accuracy(pred,y)
ber2 = BER(pred,y)

# %%
answers['Q2'] = [acc2, ber2]

# %%
assertFloatList(answers['Q2'], 2)

# %%
### Question 3

# %%
random.seed(3)
random.shuffle(dataset)

# %%
X = [d[:-1] for d in dataset]
y = [d[-1] for d in dataset]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
len(Xtrain), len(Xvalid), len(Xtest)

# %%
mod = linear_model.LogisticRegression(C=1, class_weight='balanced')
mod.fit(Xtrain,ytrain)

pred = mod.predict(Xtrain)
berTrain = BER(pred, ytrain)

# %%
mod.fit(Xvalid, yvalid)
pred = mod.predict(Xvalid)
berValid = BER(pred, yvalid)

# %%
mod.fit(Xtest, ytest)
pred = mod.predict(Xtest)
berTest = BER(pred, ytest)

# %%
answers['Q3'] = [berTrain, berValid, berTest]

# %%
assertFloatList(answers['Q3'], 3)

answers['Q3']

# %%
### Question 4

# %%
C_values = [10 ** i for i in range (-4, 5)] # from 10^-4 to 10^4
berList = []
for C in C_values:
    model = linear_model.LogisticRegression(C=C, class_weight = 'balanced')
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xvalid)
    berList.append(BER(pred, yvalid))

# %%
answers['Q4'] = berList
berList

# %%
assertFloatList(answers['Q4'], 9)

# %%
### Question 5

# %%
ber5 = min(berList)
bestC = C_values[berList.index(ber5)]


# %%
answers['Q5'] = [bestC, ber5]

answers['Q5']

# %%
assertFloatList(answers['Q5'], 2)

# %%
### Question 6

# %%
f = gzip.open("young_adult_10000.json.gz")
dataset = []
for l in f:
    dataset.append(eval(l))

# %%
dataTrain = dataset[:9000]
dataTest = dataset[9000:]
dataTrain[0]

# %%
# Some data structures you might want

usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in dataTrain:
    user = d["user_id"]
    item = d["book_id"]
    rating = d["rating"]

    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)

    ratingDict[(user, item)] = rating

# %%
def Jaccard(s1, s2):
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union else 0

# %%
def mostSimilar(i, N):
    simlilarity = []

    # set of users who rated item i
    target_users = usersPerItem[i]

    for item, users in usersPerItem.items():
        if item != i:
            sim = Jaccard(target_users, users)
            simlilarity.append((sim, item))

    most_simiar_items = sorted(simlilarity, reverse=True)[:N]
    return most_simiar_items

# %%
answers['Q6'] = mostSimilar('2767052', 10)

# %%
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)

# %%
### Question 7

# %% [markdown]
# 

# %%
average_ratings = {item: sum([review['rating'] for review in reviews]) / len(reviews) for item, reviews in reviewsPerItem.items()}

def predict_rating(user, item):
    if item not in average_ratings:
        return sum(average_ratings.values()) / len(average_ratings)
    numerator, denominator = 0, 0

    # Loop over items the user has rated
    for j in itemsPerUser[user] - {item}:
        if j in average_ratings:  # Only consider items with an average rating
            sim = Jaccard(usersPerItem[item], usersPerItem[j])
            numerator += (ratingDict[(user, j)] - average_ratings[j]) * sim
            denominator += sim
    
    # the final predicted rating
    if denominator == 0:
        return average_ratings[item]  # If no similar items, use item's average rating
    else:
        return average_ratings[item] + numerator / denominator
    

# %%
def MSE(data):
    mse = 0
    for d in data:
        user, item, actual_rating = d["user_id"], d["book_id"], d["rating"]
        predicted_rating = predict_rating(user, item)
        mse += (predicted_rating - actual_rating) ** 2
    return mse / len(data)

# %%
mse7 = MSE(dataTest)
answers['Q7'] = mse7

# %%
assertFloat(answers['Q7'])

# %%
### Question 8

# %%
average_ratings_user = {user: sum([review['rating'] for review in reviews]) / len(reviews) for user, reviews in reviewsPerUser.items()}

def predict_rating(u, i):
    if u not in average_ratings_user:
        return sum(average_ratings_user.values()) / len(average_ratings_user)
    
    numerator, denominator = 0, 0

    for v in usersPerItem[item] - {user}:
        if v in average_ratings_user:  # Only consider items with an average rating
            sim = Jaccard(itemsPerUser[user], itemsPerUser[v])
            numerator += (ratingDict[(v, item)] - average_ratings_user[v]) * sim
            denominator += sim
    
    # final calculation
    if denominator == 0:
        return average_ratings_user[user]  # If no similar items, use item's average rating
    else:
        return average_ratings_user[user] + numerator / denominator
    

# %%
mse8 = MSE(dataTest)
answers['Q8'] = mse8

# %%
assertFloat(answers['Q8'])

# %%
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')
f.close()


