# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import random
import pandas as pd

# %%
# Process data
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

# Load data
train_data = list(readCSV("train_Interactions.csv.gz"))
test_data = pd.read_csv("pairs_Read.csv")

# %%
# Read Prediction Task

# %%
# Define the Jaccard similarity function
def jaccard_similarity(users1, users2):
    intersection = len(users1 & users2)
    union = len(users1 | users2)
    return intersection / union if union else 0

# Predict using a hybrid model (popularity + similarity)
def predict_read(test_data, ratingsPerUser, ratingsPerItem, sim_threshold, pop_threshold):
    predictions = []
    for _, row in test_data.iterrows():
        user, book = row['userID'], row['bookID']
        max_similarity = 0

        # Compute maximum Jaccard similarity
        if user in ratingsPerUser and book in ratingsPerItem:
            target_users = set(ratingsPerItem[book])
            for b_read, _ in ratingsPerUser[user]:
                read_users = set(ratingsPerItem[b_read])
                similarity = jaccard_similarity(target_users, read_users)
                max_similarity = max(max_similarity, similarity)
        
        # Determine prediction based on thresholds
        pred = 0
        if max_similarity > sim_threshold or len(ratingsPerItem.get(book, [])) > pop_threshold:
            pred = 1
        predictions.append((user, book, pred))
    return predictions

# Prepare user and item ratings mappings
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u, b, r in train_data:
    ratingsPerUser[u].append((b, r))
    ratingsPerItem[b].append((u, r))

# Parameters
similarity_threshold = 0.027  # Jaccard similarity threshold
popularity_threshold = 30    # Popularity threshold

# Predict for test data
predictions = predict_read(
    test_data,
    ratingsPerUser,
    ratingsPerItem,
    similarity_threshold,
    popularity_threshold
)

# Save predictions to CSV
output = pd.DataFrame(predictions, columns=['userID', 'bookID', 'prediction'])
output.to_csv('predictions_Read.csv', index=False)

# %%
# Rating Prediction Task

# %%
# Load data
train_data = pd.read_csv('train_Interactions.csv.gz')
test_data = pd.read_csv('pairs_Rating.csv')

# Map user and book IDs to indices
user_ids = {u: i for i, u in enumerate(set(train_data['userID']))}
book_ids = {b: i for i, b in enumerate(set(train_data['bookID']))}
num_users = len(user_ids)
num_books = len(book_ids)

# Initialize parameters
num_factors = 1
learning_rate = 0.005
lambda_reg = 0.3  # Regularization parameter
num_iterations = 50
early_stopping_tolerance = 1e-4

# Global average
global_avg = train_data['rating'].mean()

# Biases and latent factors
user_bias = np.zeros(num_users)
item_bias = np.zeros(num_books)
P = np.random.normal(scale=0.05, size=(num_users, num_factors))  # User latent factors
Q = np.random.normal(scale=0.05, size=(num_books, num_factors))  # Item latent factors

# Split training data into training and validation sets
train, valid = train_test_split(train_data, test_size=0.1, random_state=42)

# Train the model using SGD with early stopping
best_val_mse = float('inf')
best_P, best_Q, best_user_bias, best_item_bias = None, None, None, None

for iteration in range(num_iterations):
    # Training phase
    for _, row in train.iterrows():
        user_idx = user_ids[row['userID']]
        book_idx = book_ids[row['bookID']]
        actual_rating = row['rating']

        # Predict the rating
        pred_rating = (global_avg + user_bias[user_idx] + item_bias[book_idx] +
                       np.dot(P[user_idx], Q[book_idx]))
        error = actual_rating - pred_rating

        # Update parameters
        user_bias[user_idx] += learning_rate * (error - lambda_reg * user_bias[user_idx])
        item_bias[book_idx] += learning_rate * (error - lambda_reg * item_bias[book_idx])
        P[user_idx] += learning_rate * (error * Q[book_idx] - lambda_reg * P[user_idx])
        Q[book_idx] += learning_rate * (error * P[user_idx] - lambda_reg * Q[book_idx])

    # Calculate training MSE
    train_mse = 0
    for _, row in train.iterrows():
        user_idx = user_ids[row['userID']]
        book_idx = book_ids[row['bookID']]
        pred_rating = (global_avg + user_bias[user_idx] + item_bias[book_idx] +
                       np.dot(P[user_idx], Q[book_idx]))
        train_mse += (row['rating'] - pred_rating) ** 2
    train_mse /= len(train)

    # Validation phase
    val_mse = 0
    for _, row in valid.iterrows():
        user_idx = user_ids[row['userID']]
        book_idx = book_ids[row['bookID']]
        pred_rating = (global_avg + user_bias[user_idx] + item_bias[book_idx] +
                       np.dot(P[user_idx], Q[book_idx]))
        val_mse += (row['rating'] - pred_rating) ** 2
    val_mse /= len(valid)

    print(f"Iteration {iteration + 1}/{num_iterations}, Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")

    # Early stopping
    if val_mse < best_val_mse:
        best_val_mse = val_mse
        best_P = P.copy()
        best_Q = Q.copy()
        best_user_bias = user_bias.copy()
        best_item_bias = item_bias.copy()
    elif best_val_mse - val_mse < early_stopping_tolerance:
        print("Early stopping triggered.")
        break

# Use the best parameters
P, Q, user_bias, item_bias = best_P, best_Q, best_user_bias, best_item_bias

# Predict for test data
predictions = []
for _, row in test_data.iterrows():
    user = row['userID']
    book = row['bookID']
    if user in user_ids and book in book_ids:
        user_idx = user_ids[user]
        book_idx = book_ids[book]
        pred_rating = (global_avg + user_bias[user_idx] + item_bias[book_idx] +
                       np.dot(P[user_idx], Q[book_idx]))
    else:
        pred_rating = global_avg  # Default for unseen users/items
    predictions.append((user, book, pred_rating))

# Save predictions to CSV
output = pd.DataFrame(predictions, columns=['userID', 'bookID', 'prediction'])
output.to_csv('predictions_Rating.csv', index=False)


