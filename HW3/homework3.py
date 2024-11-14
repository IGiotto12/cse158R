# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

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
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# %%
answers = {}

# %%
# Some data structures that will be useful

# %%
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

# %%
len(allRatings)

# %%
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))

ratingsValid[0]

# %%
##################################################
# Read prediction                                #
##################################################

# %%
# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort(reverse=True)

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

# %%
### Question 1

# %%
validation_set = []
for user, book, _ in ratingsValid:
    # Add the positive sample
    validation_set.append((user, book, 1))  # 1 indicates a read
    
    # Generate a negative sample (a book the user hasn’t read)
    while True:
        negative_book = random.choice(list(bookCount.keys()))
        if negative_book not in [b for b, _ in ratingsPerUser[user]]:
            validation_set.append((user, negative_book, 0))  # 0 indicates non-read
            break

# Evaluate the baseline model on this validation set
correct_predictions = 0
for user, book, actual in validation_set:
    prediction = 1 if book in return1 else 0
    if prediction == actual:
        correct_predictions += 1

acc1 = correct_predictions / len(validation_set)

# %%
answers['Q1'] = acc1
assertFloat(answers['Q1'])

# %%
### Question 2

# %%
threshold = 0
acc2 = 0

# Loop through different thresholds
for threshold_percentage in range(20, 80, 10):
    threshold_value = (threshold_percentage / 100) * totalRead
    current_return1 = set()
    count = 0
    
    for ic, i in mostPopular:
        count += ic
        current_return1.add(i)
        if count > threshold_value:
            break
    
    # Calculate accuracy on the validation set
    correct_predictions = 0
    for user, book, actual in validation_set:
        prediction = 1 if book in current_return1 else 0
        if prediction == actual:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(validation_set)
    
    # Update best threshold if current accuracy is higher
    if accuracy > acc2:
        acc2 = accuracy
        threshold = threshold_percentage

# %%
answers['Q2'] = [threshold, acc2]

assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])

# %%
### Question 3/4

# %%
# Define the Jaccard similarity function
def jaccard_similarity(book1, book2):
    users1 = set(u for u, _ in ratingsPerItem[book1])
    users2 = set(u for u, _ in ratingsPerItem[book2])
    intersection = len(users1 & users2)
    union = len(users1 | users2)
    return intersection / union if union else 0

# Construct the validation set with clear labels
validation_set = []
for user, book, _ in ratingsValid:
    # Add the positive sample
    validation_set.append((user, book, 1))  # 1 indicates a read (positive sample)
    
    # Generate a negative sample (a book the user hasn’t read)
    while True:
        negative_book = random.choice(list(bookCount.keys()))
        if negative_book not in [b for b, _ in ratingsPerUser[user]]:
            validation_set.append((user, negative_book, 0))  # 0 indicates non-read (negative sample)
            break

# Precompute maximum Jaccard similarity for each (user, book) pair in the validation set
jaccard_similarities = {}

for user, book, actual in validation_set:
    max_similarity = 0
    
    # Calculate the maximum Jaccard similarity with books the user has read
    for b, _ in ratingsPerUser[user]:
        similarity = jaccard_similarity(book, b)
        max_similarity = max(max_similarity, similarity)
    
    # Store the max similarity for this (user, book) pair
    jaccard_similarities[(user, book)] = max_similarity

# Optimize the threshold for maximum accuracy
best_jaccard_threshold = 0
acc3 = 0

# Loop through a range of threshold values to find the best threshold
for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    correct_predictions = 0
    
    for user, book, actual in validation_set:
        max_similarity = jaccard_similarities[(user, book)]
        
        # Predict "read" if max similarity exceeds the threshold
        prediction = 1 if max_similarity >= threshold else 0
        if prediction == actual:
            correct_predictions += 1
    
    # Calculate the accuracy for the current threshold
    accuracy = correct_predictions / len(validation_set)
    
    # Update best threshold if current accuracy is higher
    if accuracy > acc3:
        acc3 = accuracy
        best_jaccard_threshold = threshold

# %%
acc4 = 0
best_combined_jaccard_threshold = 0
best_combined_popularity_threshold = 0

# Loop through possible Jaccard thresholds
for jaccard_threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for pop_threshold_percentage in range(20, 80, 10):
        threshold_value = (pop_threshold_percentage / 100) * totalRead
        current_return1 = set()
        count = 0
        
        for ic, i in mostPopular:
            count += ic
            current_return1.add(i)
            if count > threshold_value:
                break
        
        # Evaluate accuracy with the combined approach
        correct_predictions = 0
        for user, book, actual in validation_set:
            is_popular = book in current_return1
            
            # Check max Jaccard similarity
            max_similarity = 0
            for b, _ in ratingsPerUser[user]:
                similarity = jaccard_similarity(book, b)
                max_similarity = max(max_similarity, similarity)
            
            # Predict read if either condition is met
            prediction = 1 if is_popular or max_similarity >= jaccard_threshold else 0
            if prediction == actual:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(validation_set)
        
        # Update best parameters if accuracy improves
        if accuracy > acc4:
            acc4 = accuracy
            best_combined_jaccard_threshold = jaccard_threshold
            best_combined_popularity_threshold = pop_threshold_percentage

# %%
answers['Q3'] = acc3
answers['Q4'] = acc4

assertFloat(answers['Q3'])
assertFloat(answers['Q4'])

# %%
predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    
    # Check if the book is popular enough (based on best popularity threshold)
    is_popular = b in return1
    
    # Check Jaccard similarity with books the user has read
    max_similarity = 0
    if u in ratingsPerUser:
        for b_read, _ in ratingsPerUser[u]:
            similarity = jaccard_similarity(b, b_read)
            max_similarity = max(max_similarity, similarity)
    
    # Predict "read" based on combined model or best individual model
    if is_popular or max_similarity >= best_combined_jaccard_threshold:
        prediction = 1  # Predict "read"
    else:
        prediction = 0  # Predict "non-read"
    
    # Write the prediction result to the output file
    predictions.write(f"{u},{b},{prediction}\n")

predictions.close()

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
assert type(answers['Q5']) == str

# %%
##################################################
# Rating prediction                              #
##################################################

# %%
### Question 6

# %%
# Initialize parameters for rating(u, i)
alpha = sum([r for _, _, r in ratingsTrain]) / len(ratingsTrain)  # Global average rating

lambda_reg = 1
beta_user = defaultdict(float)
beta_item = defaultdict(float)

# %%
# Define gradient descent function for bias terms
def gradient_descent(lr=0.0001, iterations=100):
    for _ in range(iterations):
        user_grad = defaultdict(float)
        item_grad = defaultdict(float)

        # Accumulate gradients for each user and item
        for u, i, r in ratingsTrain:
            prediction = alpha + beta_user[u] + beta_item[i]
            error = r - prediction
            
            # Gradient calculation with regularization
            user_grad[u] += -2 * error + 2 * lambda_reg * beta_user[u]
            item_grad[i] += -2 * error + 2 * lambda_reg * beta_item[i]
        
        # Update bias terms with a cap to avoid large values
        for u in user_grad:
            beta_user[u] = max(min(beta_user[u] - lr * user_grad[u], 10), -10)
        for i in item_grad:
            beta_item[i] = max(min(beta_item[i] - lr * item_grad[i], 10), -10)

# Run gradient descent with adjusted parameters
gradient_descent()

# Calculate MSE on validation set with controlled values
validMSE = 0
for u, i, r in ratingsValid:
    prediction = alpha + beta_user[u] + beta_item[i]
    validMSE += (r - prediction) ** 2

validMSE /= len(ratingsValid) if len(ratingsValid) else 0


# %%
answers['Q6'] = validMSE
assertFloat(answers['Q6'])

# %%
### Question 7

# %%
# Find the user with the largest beta_user value
maxUser = max(beta_user, key=beta_user.get)
maxBeta = beta_user[maxUser]

# Find the user with the smallest (most negative) beta_user value
minUser = min(beta_user, key=beta_user.get)
minBeta = beta_user[minUser]

# %%
answers['Q7'] = [maxUser, minUser, maxBeta, minBeta]
assert [type(x) for x in answers['Q7']] == [str, str, float, float]

# %%
### Question 8

# %%
# Define a range of lambda values
lambda_values = [0.1, 0.5, 1, 5, 10]
lamb = None
validMSE = float('inf')

# Initialize global average rating alpha based on training data
alpha = sum([r for _, _, r in ratingsTrain]) / len(ratingsTrain)

# Function to calculate MSE for a given lambda with regularization
def calculate_mse_for_lambda(lambda_reg, lr=0.005, max_iter=100, tol=1e-4):
    # Initialize bias terms
    beta_user = defaultdict(float)
    beta_item = defaultdict(float)
    
    # Gradient descent with decaying learning rate and convergence check
    for iteration in range(max_iter):
        user_grad = defaultdict(float)
        item_grad = defaultdict(float)
        
        # Accumulate gradients for each user and item with regularization
        for u, i, r in ratingsTrain:
            prediction = alpha + beta_user[u] + beta_item[i]
            error = r - prediction
            
            # Gradient calculation with L2 regularization
            user_grad[u] += -2 * error + 2 * lambda_reg * beta_user[u]
            item_grad[i] += -2 * error + 2 * lambda_reg * beta_item[i]
        
        # Update beta_user and beta_item with capped values
        learning_rate = lr / (1 + 0.01 * iteration)
        max_change = 0
        
        for u in user_grad:
            update = learning_rate * user_grad[u]
            beta_user[u] = max(min(beta_user[u] - update, 10), -10)  # Cap to [-10, 10]
            max_change = max(max_change, abs(update))
        
        for i in item_grad:
            update = learning_rate * item_grad[i]
            beta_item[i] = max(min(beta_item[i] - update, 10), -10)  # Cap to [-10, 10]
            max_change = max(max_change, abs(update))
        
        # Stop if updates are very small
        if max_change < tol:
            break

    # Calculate MSE on validation set
    mse = 0
    for u, i, r in ratingsValid:
        prediction = alpha + beta_user[u] + beta_item[i]
        mse += (r - prediction) ** 2
    
    # Regularization term
    reg_term = lambda_reg * (sum(v ** 2 for v in beta_user.values()) + sum(v ** 2 for v in beta_item.values()))
    
    # Final MSE with regularization
    mse = mse / len(ratingsValid) + reg_term
    
    return mse

# Loop through lambda values to find the best one based on MSE
for lambda_reg in lambda_values:
    mse = calculate_mse_for_lambda(lambda_reg)
    if mse < validMSE:
        validMSE = mse
        lamb = lambda_reg

# %%
answers['Q8'] = (lamb, validMSE)

assertFloat(answers['Q8'][0])
assertFloat(answers['Q8'][1])

# %%
predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        predictions.write(l)
        continue
    u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
    # Prediction using the learned biases
    prediction = alpha
    if u in beta_user:
        prediction += beta_user[u]
    if b in beta_item:
        prediction += beta_item[b]
    
    # Write the prediction to the file
    predictions.write(f"{u},{b},{prediction}\n")
    
predictions.close()

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


