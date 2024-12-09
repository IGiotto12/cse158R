# %%
import gzip
import json
import csv

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

# %%
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        b,u,r,t = l.strip().split(',')
        r = int(r)
        yield b,u,r,t

# %%
allRatings = []
for l in readCSV("rating-Alaska.csv.gz"):
    allRatings.append(l)

allRatings

# %% [markdown]
# ### JSON TO CSV

# %%
input_file = 'review-Alaska_10.json'
output_file = 'review-Alaska_10.csv'

with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    with open(input_file, 'r', encoding='utf-8') as json_file:
        first_line = True
        for line in json_file:
            json_object = json.loads(line.strip())
            
            if first_line:
                csv_writer.writerow(json_object.keys())
                first_line = False
            
            csv_writer.writerow(json_object.values())

print(f"Data successfully converted to {output_file}")

# %%
input_file = 'review-Alaska_10.json'

with open(input_file, 'r', encoding='utf-8') as json_file:
    data = [json.loads(line.strip()) for line in json_file]

# %%
data

# %%
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import metrics
import math

# %%
df = pd.read_csv('review-Alaska_10.csv')

# %% [markdown]
# #### Word Cloud Visualization E.g.

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df5 = df[(df['rating'] == 5) & (df['text'].notna())]

text5 = " ".join(df5['text'])

# Generate and display the word cloud
wordcloud5 = WordCloud(background_color="white").generate(text5)
plt.imshow(wordcloud5)
plt.axis("off")
plt.show()

# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt

df1 = df[(df['rating'] == 1) & (df['text'].notna())]

text1 = " ".join(df1['text'])

wordcloud1 = WordCloud(background_color="white").generate(text1)
plt.imshow(wordcloud1)
plt.axis("off")
plt.show()

# %%


# %%
df = df[df['text'].notna()]

# %%
df = df.sample(frac=0.2)

# %%
x = df['text']
y = df['rating']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state = 2022)

vectorizer = CountVectorizer(ngram_range = (1, 2), min_df=10)

vectorizer.fit(x_train)
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

X_train.toarray()
X_test.toarray()
Y_train = np.array(y_train)
Y_test = np.array(y_test)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)

# %%
Y_pred = regr.predict(X_test)
ssr = np.sum(Y_pred - np.mean(Y_test)) ** 2
mse = metrics.mean_squared_error(Y_test, Y_pred)
rmse = math.sqrt(mse)
print('R^2 score: %.2f' % regr.score(X_test, Y_test))
print('RMSE: %.2f' % rmse)

# %%
df_features = pd.DataFrame(data = {'Coefficient':list(regr.coef_),'Feature_Name':vectorizer.get_feature_names_out()})
df_features['Coefficient_Magnitude'] = abs(df_features['Coefficient'])
df_features.sort_values(by='Coefficient_Magnitude', ascending=False).head(10)

# %%


# %% [markdown]
# ### Lasso

# %%
alpha_list = [0.0001, 0.001, 0.01, 0.1]
col_labels_lasso = ['Alpha', 'Training RMSE', 'Model Complexity - Coef Norm1', 'Model Complexity - Coef Sum', 'Test RMSE']

result_lasso_arr = []
for alpha in alpha_list:
    result_lasso_list=[]
    #build model:
    lasso = linear_model.Lasso(alpha)
    lasso.fit(X_train, Y_train)
    #applied to test data
    Y_pred_test = lasso.predict(X_test)
    mse_test = metrics.mean_squared_error(Y_test, Y_pred_test)
    rmse_test = math.sqrt(mse_test)
    #applied to train data
    Y_pred_train = lasso.predict(X_train)
    mse_train = metrics.mean_squared_error(Y_train, Y_pred_train)
    rmse_train = math.sqrt(mse_train)
    #compute complexity by L1-norm of the model parameter values
    complexity_coef_norm1 = np.linalg.norm(lasso.coef_, ord=1)
    #compute complexity by sum of the model parameter magnitudes
    complexity_coef_sum = np.sum(np.abs(lasso.coef_))
    #output result
    print(f'Alpha value: {alpha}')
    print(f'Train RMSE: {rmse_train}')
    print(f'Model Complexity - Norm1 of Coefficients: {complexity_coef_norm1}')
    print(f'Model Complexity - Sum of Coefficients: {complexity_coef_sum}')
    print(f'Test RMSE: {rmse_test}\n')
    result_lasso_list = [alpha, rmse_train, complexity_coef_norm1, complexity_coef_sum, rmse_test]
    result_lasso_arr.append(result_lasso_list)
    
df_lasso = pd.DataFrame(result_lasso_arr,columns=col_labels_lasso)
df_lasso

# %%
lasso_best = linear_model.Lasso(0.0001)
lasso_best.fit(X_train, Y_train)
df_lasso_features = pd.DataFrame(data = {'Coefficient':list(lasso_best.coef_),'Feature_Name':vectorizer.get_feature_names_out()})
df_lasso_features['Coefficient_Magnitude'] = abs(df_lasso_features['Coefficient'])

# %%
df_lasso_features.sort_values(by='Coefficient_Magnitude', ascending=False).head(100)

# %% [markdown]
# ### Ridge

# %%
alpha_list = [0.0001, 0.001, 0.01, 0.1]
col_labels_ridge = ['Alpha', 'Training RMSE', 'Model Complexity - Coef Norm2', 'Model Complexity - Coef Sum', 'Test RMSE']

result_ridge_arr = []
for alpha in alpha_list:
    result_ridge_list=[]
    #build model:
    ridge = linear_model.Ridge(alpha)
    ridge.fit(X_train, Y_train)
    #applied to test data
    Y_pred_test = ridge.predict(X_test)
    mse_test = metrics.mean_squared_error(Y_test, Y_pred_test)
    rmse_test = math.sqrt(mse_test)
    #applied to train data
    Y_pred_train = ridge.predict(X_train)
    mse_train = metrics.mean_squared_error(Y_train, Y_pred_train)
    rmse_train = math.sqrt(mse_train)
    #compute complexity by L1-norm of the model parameter values
    complexity_coef_norm2 = np.linalg.norm(ridge.coef_, ord=2)
    #compute complexity by sum of the model parameter magnitude
    complexity_coef_sum = np.sum(np.abs(ridge.coef_))
    #output result
    print(f'Alpha value: {alpha}')
    print(f'Train RMSE: {rmse_train}')
    print(f'Model Complexity - Norm2 of Coefficients: {complexity_coef_norm2}')
    print(f'Model Complexity - Sum of Coefficients: {complexity_coef_sum}')
    print(f'Test RMSE: {rmse_test}\n')
    result_ridge_list = [alpha, rmse_train, complexity_coef_norm2, complexity_coef_sum, rmse_test]
    result_ridge_arr.append(result_ridge_list)

df_ridge = pd.DataFrame(result_ridge_arr,columns=col_labels_ridge)
df_ridge

# %%
ridge_best = linear_model.Ridge(0.1)
ridge_best.fit(X_train, Y_train)
df_ridge_features = pd.DataFrame(data = {'Coefficient':list(ridge_best.coef_),'Feature_Name':vectorizer.get_feature_names_out()})
df_ridge_features['Coefficient_Magnitude'] = abs(df_ridge_features['Coefficient'])
df_ridge_features.sort_values(by='Coefficient_Magnitude', ascending=False).head(10)

# %% [markdown]
# ### Regularized Logistic Regression

# %%
df_1_and_5 = df[(df.rating == 5) | (df.rating == 1)]

x = df_1_and_5['text']
y = df_1_and_5['rating']
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state = 2022)

vectorizer = CountVectorizer(ngram_range = (1, 2), min_df=10)
X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.fit_transform(x_test)

vectorizer.fit(x)
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)
Y_train = np.array(y_train)
Y_test = np.array(y_test)

# %% [markdown]
# #### L2

# %%
C_list = [0.001, 0.01, 0.1, 1, 10, 100] 
col_labels = ['C', 'Train AUC','Model Complexity - Coef Sum','Test AUC']

result_log_L2_arr = []
result_fpr_L2_arr = []
result_tpr_L2_arr = []
for c_value in C_list:
    result_log_L2_list = []
    
    # Build the logistic regression estimator with the specified C value
    estimator_L2 = linear_model.LogisticRegression(solver = 'liblinear', max_iter=1000, random_state=2022, C=c_value,penalty='l2') 
    estimator_L2.fit(X_train, Y_train)
    # Check the class order, make sure 5 is on the 2nd column
    print('Class order:',estimator_L2.classes_)
    if estimator_L2.classes_[1] == 5:
        print('Correct positive class selection')
    else:
        print('Incorrect positive class selection, check')
    
    # Calculate AUC_train by applying the estimator on X_train
    Y_train_pred_prob = estimator_L2.predict_proba(X_train)
    Y_train_pred_prob_5star = Y_train_pred_prob[:, 1]
    AUC_train = metrics.roc_auc_score(Y_train, Y_train_pred_prob_5star)
    
    # Calculate AUC_test by applying the estimator on X_test
    Y_test_pred_prob = estimator_L2.predict_proba(X_test)
    Y_test_pred_prob_5star = Y_test_pred_prob[:, 1]
    AUC_test = metrics.roc_auc_score(Y_test, Y_test_pred_prob_5star)
    
    # Calculate fpr and tpr from test data
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_pred_prob_5star, pos_label=5)
    
    # Compute complexity by sum of the model parameter magnitudes
    complexity_coef_sum = np.sum(np.abs(estimator_L2.coef_))
    
    # Compile results
    result_log_L2_list = [c_value, AUC_train, complexity_coef_sum, AUC_test]
    result_log_L2_arr.append(result_log_L2_list)
    result_fpr_L2_arr.append(list(fpr))
    result_tpr_L2_arr.append(list(tpr))
    
    print('C value:',c_value)
    print('Model Complexity:',complexity_coef_sum)
    print('Train_AUC:', AUC_train)
    print('Test_AUC:', AUC_test,'\n')

df_log_L2 = pd.DataFrame(result_log_L2_arr,columns=col_labels)
df_log_L2

# %%
colours=['r','g','b','k','m','c']
plt.figure(figsize = (8, 8))
for i in range(len(result_fpr_L2_arr)):
    plt.plot(result_fpr_L2_arr[i],result_tpr_L2_arr[i],colours[i])
plt.gca().legend(df_log_L2['Test AUC'])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# %%
estimator_L2_best = linear_model.LogisticRegression(solver = 'liblinear', max_iter=1000, random_state=2022, C=0.01,penalty='l2') 
estimator_L2_best.fit(X_train, Y_train)

df_L2_features = pd.DataFrame(data = {'Coefficient':estimator_L2_best.coef_.tolist()[0],'Feature_Name':vectorizer.get_feature_names_out()})
df_L2_features['Coefficient_Magnitude'] = abs(df_L2_features['Coefficient'])
df_L2_features.sort_values(by='Coefficient_Magnitude', ascending=False).head(10)

# %% [markdown]
# #### L1

# %%
C_list = [0.001, 0.01, 0.1, 1, 10, 100] 
col_labels = ['C', 'Train AUC','Model Complexity - Coef Sum','Test AUC']

result_log_L1_arr = []
result_fpr_L1_arr = []
result_tpr_L1_arr = []
for c_value in C_list:
    result_log_L1_list = []
    
    # Build the logistic regression estimator with the specified C value
    estimator_L1 = linear_model.LogisticRegression(solver = 'liblinear', max_iter=1000, random_state=2022, C=c_value,penalty='l1') 
    estimator_L1.fit(X_train, Y_train)
    # Check the class order, make sure 5 is on the 2nd column
    print('Class order:',estimator_L1.classes_)
    if estimator_L1.classes_[1] == 5:
        print('Correct positive class selection')
    else:
        print('Incorrect positive class selection, check')
    
    # Calculate AUC_train by applying the estimator on X_train
    Y_train_pred_prob = estimator_L1.predict_proba(X_train)
    Y_train_pred_prob_5star = Y_train_pred_prob[:, 1]
    AUC_train = metrics.roc_auc_score(Y_train, Y_train_pred_prob_5star)
    
    # Calculate AUC_test by applying the estimator on X_test
    Y_test_pred_prob = estimator_L1.predict_proba(X_test)
    Y_test_pred_prob_5star = Y_test_pred_prob[:, 1]
    AUC_test = metrics.roc_auc_score(Y_test, Y_test_pred_prob_5star)
    
    # Calculate fpr and tpr from test data
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_pred_prob_5star, pos_label=5)
    
    # Compute complexity by sum of the model coefficients' magnitude
    complexity_coef_sum = np.sum(np.abs(estimator_L1.coef_))
    
    # Compile results
    result_log_L1_list = [c_value, AUC_train, complexity_coef_sum, AUC_test]
    result_log_L1_arr.append(result_log_L1_list)
    result_fpr_L1_arr.append(list(fpr))
    result_tpr_L1_arr.append(list(tpr))
    
    print('C value:',c_value)
    print('Model Complexity:',complexity_coef_sum)
    print('Train_AUC:', AUC_train)
    print('Test_AUC:', AUC_test,'\n')
    
df_log_L1 = pd.DataFrame(result_log_L1_arr,columns=col_labels)
df_log_L1

# %%
estimator_L1_best = linear_model.LogisticRegression(solver = 'liblinear', max_iter=1000, random_state=2022, C=0.1,penalty='l1') 
estimator_L1_best.fit(X_train, Y_train)

# Obtain the features and their magnitude, then sorted by magnitide
df_L1_features = pd.DataFrame(data = {'Coefficient':estimator_L1_best.coef_.tolist()[0],'Feature_Name':vectorizer.get_feature_names_out()})
df_L1_features['Coefficient_Magnitude'] = abs(df_L1_features['Coefficient'])
df_L1_features.sort_values(by='Coefficient_Magnitude', ascending=False).head(10)

# %%



