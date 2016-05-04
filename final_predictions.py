from __future__ import print_function
from sys import argv, exit
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

N_ESTIMATORS = 40

# Helper Functions

def select_features(df):
    zero_variance_cols = ['29', '31', '32', '35']
    redundant_cols = ['18', '23', '25', '26', '58']

    cols_to_drop = list(set(zero_variance_cols + redundant_cols))
    df.drop(cols_to_drop, axis=1, inplace=True)

def one_hot(df):
    # categorical columns
    categorical_cols = ['0', '5', '7', '8', '9', '14', '16', '17', '20', '56', '57']
    # numerical columns with [0,1,2] values only
    num_to_categorical_cols = [str(i) for i in range(38,52)]
    categorical_cols_enhanced = categorical_cols + num_to_categorical_cols

    # convert categorical to one-hot sparse representation
    df_one_hot = pd.get_dummies(df, columns=categorical_cols_enhanced)

    return df_one_hot

# For quiz data we only want to include features contained
# in the trainig data
def one_hot_quiz(df_quiz, train_columns):
    df_one_hot_quiz = one_hot(df_quiz)

    # inflate quiz data with complete one-hot representation
    col_to_add = np.setdiff1d(train_columns, df_one_hot_quiz.columns)
    for c in col_to_add:
        df_one_hot_quiz[c] = 0

    df_one_hot_quiz = df_one_hot_quiz[train_columns]

    return df_one_hot_quiz

# Main functions

def learn(datafile):
    # Read training data
    df_train = pd.read_csv(datafile)

    # Separate training labels from features
    df_y = df_train['label'].copy()
    del df_train['label']

    # Drop specific features
    select_features(df_train)

    # One-hot encoding for categorical variables
    df_train = one_hot(df_train)

    rfc = RandomForestClassifier(n_estimators = N_ESTIMATORS)
    rfc.fit(df_train, df_y)

    return rfc, df_train.columns

def predict(quizfile, rfc, train_columns):
    # Read quiz data
    df_quiz = pd.read_csv(quizfile)

    # Drop specific features
    select_features(df_quiz)

    # One-hot encoding for categorical variables
    df_quiz = one_hot_quiz(df_quiz, train_columns)

    quiz_preds = rfc.predict(df_quiz)

    return quiz_preds

def writeoutput(outputfile, preds):
    with open(outputfile, 'w') as f:
        f.write("Id,Prediction\n")
        for i, pred in enumerate(preds):
            f.write("%d,%d\n" % (i+1,pred))

# Main

def main(datafile, quizfile, outputfile):
    rfc, train_columns = learn(datafile)
    quiz_preds = predict(quizfile, rfc, train_columns)
    writeoutput(outputfile, quiz_preds)


if __name__ == '__main__':
    if len(argv) != 4:
        print ("Usage: python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE")
        exit(1)
    else:
        datafile, quizfile, outputfile = argv[1:4]
        main(datafile, quizfile, outputfile)
