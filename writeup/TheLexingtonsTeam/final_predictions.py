# Felix Matathias
# Swapnil Khedekar
# Craig Liebman
#
# Kaggle team: TheLexingtons
#
# Running time: under a minute

import pandas as pd
import numpy as np
from sys import argv, exit
from sklearn.ensemble import RandomForestClassifier

data_file = None
quiz_file = None
out_file  = None 

if len(argv) != 4:
    print ("Usage: python final_predictions.py DATAFILE QUIZFILE OUTPUTFILE")
    exit(1)
else:
    data_file, quiz_file, output_file = argv[1:4]

# Create data frames    
df = pd.read_csv(data_file)
df_quiz = pd.read_csv(quiz_file)

# Separate labels from input vectors
df_y = df['label'].copy()
del df['label']

# Delete columns that have no variance or are linear combinations of other columns
cols_to_delete = ['18','25','29', '31', '32', '35', '23', '26', '58']  
for col in cols_to_delete:
    del df[col]
    del df_quiz[col]

# Get list of remaining columns
df_cols = list(df.columns.values)

# Holds the unique values for each column, will be used later on for inflating the quiz data
df_cols_dict = {}
for col in df_cols:
    df_cols_dict[col] = pd.value_counts(df[col].values).to_dict().keys()

# Define categorical columns
categorical_cols = ['56', '20', '14', '17', '16', '57', '0', '5', '7', '9', '8']

# Numerical columns with [0,1,2] vals, also converting to categorical
num_to_categorical_cols = [str(i) for i in range(38,52)]

# Union of above columns
categorical_cols_enhanced = categorical_cols + num_to_categorical_cols  

# Convert categorical to one-hot sparse column
df_one_hot = pd.get_dummies(df, columns=categorical_cols_enhanced)

# Prepare to inflate quiz data

# list of dicts with inflated data, will be used to construct quiz dataframe
quiz_raw_data = []

# different column types, will be inflated differently 
cols_set = set(df_cols)
cols_categ_set = set(categorical_cols_enhanced)
cols_num_set = cols_set - cols_categ_set
cols_num_to_categ = set(num_to_categorical_cols)

# Sanity check
if len(set.union(cols_num_set,cols_categ_set)) != len(cols_set):
    raise RuntimeError

# Inflate quiz data since they do not contain all the columns/values found in the training data
for i in range(len(df_quiz)):
    x = df_quiz.iloc[i].to_dict()
    x_inflated = {}
    for k,v in x.items():
        if k in cols_num_set:
            x_inflated[k] = v
        elif k in cols_num_to_categ:
            for k2 in ['0.0','1.0','2.0']:
                inflated_col = k + "_" + k2
                x_inflated[inflated_col] = 0
            x_inflated[k + "_" + str(v)] = 1
        else:
            for k2 in df_cols_dict[k]:
                inflated_col = k + "_" + k2
                if v == k2:
                    x_inflated[inflated_col] = 1.0 
                else:
                    x_inflated[inflated_col] = 0.0
                    
    quiz_raw_data.append(x_inflated)
            

df_cols_inflated = list(df_one_hot.columns.values)

# Create data frame with quiz data
df_one_hot_quiz = pd.DataFrame(data=quiz_raw_data, columns=df_cols_inflated)

# Prepare clasifier
rfc = RandomForestClassifier(random_state=1, n_estimators=40)

# Fit training data
rfc.fit(df_one_hot, df_y,)

# Predict
y_rfc_pred = rfc.predict(df_one_hot_quiz)

# Write output file
f_out = open(output_file, 'w')
f_out.write("Id,Prediction\n")
for i in range(1,len(y_rfc_pred)+1):
    f_out.write(str(i)+','+str(y_rfc_pred[i-1])+'\n')
f_out.close()

