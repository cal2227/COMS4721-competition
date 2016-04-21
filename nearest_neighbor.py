

import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score

def get_predictions(cls, test_features):
    predicted = cls.predict(test_features)
    predicted_probs = cls.predict_proba(test_features).transpose()[1]
    predicted = pd.Series(predicted, index=test_features.index)
    predicted.name = 'predicted'
    return predicted, predicted_probs



df = pd.read_csv("./data/data.csv")
df_quiz = pd.read_csv("./data/quiz.csv")

df_y = df['label'].copy()
del df['label']

# A
# We always want to delete these columns
cols_to_delete = ['18', '25', '29', '31', '32', '35', '23', '26', '58']
for col in cols_to_delete:
    del df[col]
    del df_quiz[col]

# B
# Test on just these pairs of features [0-8], [9-17]
cols_to_keep = ['0', '2', '5', '7', '8', '9', '11', '14', '16', '17']
for col in [col for col in df.columns if col not in cols_to_keep]:
    del df[col]
    del df_quiz[col]


# C
# Normalize the continuous columns (if not removed in B)
# df['59'] = (df['59'] - df['59'].mean()) /  (df['59'].max() - df['59'].min())
# df['60'] = (df['60'] - df['60'].mean()) /  (df['60'].max() - df['60'].min())

# df_quiz['59'] = (df_quiz['59'] - df_quiz['59'].mean()) /  (df_quiz['59'].max() - df_quiz['59'].min())
# df_quiz['60'] = (df_quiz['60'] - df_quiz['60'].mean()) /  (df_quiz['60'].max() - df_quiz['60'].min())


# Define categorical columns
categorical_cols_enhanced = list(df.columns)

# Remove purely numeric columns
if '59' in categorical_cols_enhanced:
    categorical_cols_enhanced.remove('59')
if '60' in categorical_cols_enhanced:
    categorical_cols_enhanced.remove('60')

# Convert categorical to one-hot sparse column
df_one_hot = pd.get_dummies(df, columns=categorical_cols_enhanced)
print ("features:", len(df_one_hot.columns))


df_one_hot_quiz = pd.get_dummies(df_quiz, columns=categorical_cols_enhanced)

col_to_add = np.setdiff1d(df_one_hot.columns, df_one_hot_quiz.columns)
for c in col_to_add:
    df_one_hot_quiz[c] = 0

df_one_hot_quiz = df_one_hot_quiz[df_one_hot.columns]

# Split data to train/test
train_size = 0.8
print ("train_size:", train_size)
X_train, X_test, y_train, y_test = train_test_split(df_one_hot, df_y, random_state=1, train_size=train_size)

knn = KNeighborsClassifier()
print (knn)

knn.fit(X_train, y_train)

test_preds, _ = get_predictions(knn, X_test)
train_preds, _ = get_predictions(knn, X_train)

print ('train accuracy:', knn.score(X_train, y_train))
print ('train precision:', precision_score(y_train, train_preds))
print ('train recall:', recall_score(y_train, train_preds))

print ('test accuracy:', knn.score(X_test, y_test))
print ('test precision:', precision_score(y_test, test_preds))
print ('test recall:', recall_score(y_test, test_preds))




# $ time python nearest_neighbor.py 
# features: 561
# train_size: ? (0.2 or 0.8)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
# train accuracy: 0.842769279293
# train precision: 0.826055312955
# train recall: 0.813911796343

# real  1m4.474s
# user  0m23.199s
# sys 0m10.770s


# $ time python nearest_neighbor.py 
# features: 561
# train_size: ? (0.2 or 0.8)
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
# train accuracy: 0.886624354476
# train precision: 0.862056984017
# train recall: 0.885123082412

# real  3m32.869s
# user  2m17.479s
# sys 0m13.738s


# $ time python nearest_neighbor.py 
# features: 284
# train_size: 0.2
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
# train accuracy: 0.911972247408
# train precision: 0.894681318681
# train recall: 0.907688191224
# test accuracy: 0.877076968562
# test precision: 0.848057766758
# test recall: 0.876638413633

# real  236m54.276s
# user  143m33.234s
# sys 1m2.499s


# Currently running on my laptop:

# $ python nearest_neighbor.py 
# features: 561
# train_size: 0.2
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')


# $ time python nearest_neighbor.py 
# features: 284
# train_size: 0.8
# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#            metric_params=None, n_jobs=1, n_neighbors=5, p=2,
#            weights='uniform')
