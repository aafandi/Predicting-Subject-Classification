import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Load csv files as dataframes, put them all in a list

data_list = []

for file in os.listdir('thesis_subjects'):
    if not file.startswith('.'):
        df = pd.read_csv('thesis_subjects/{file}'.format(file = file))
        data_list.append(df)



# Make a (master) dataframe that has all the data

master_df = pd.concat(data_list)


# Clean up the data

master_df = master_df[master_df['thesis'] != '\n\n']
master_df = master_df.drop_duplicates(subset='thesis')
master_df['thesis'] = master_df['thesis'].str.replace('\n\n', '')
master_df['subject'] = master_df['subject'].str.replace('Mathematics Subject Classification: ', '')


# Instantiate CountVectorizer class

count_vect = CountVectorizer()


# Make a sparse NumPy array. Each row corresponds to a thesis title, each column
# corresponds to a word in the corpus. The (i, j) element in the array is the number of
# of times thesis i has the word j in its title

X_train_counts = count_vect.fit_transform(master_df['thesis'])


# Associate (or rather, fit/train) each title to its classification

clf = MultinomialNB().fit(X_train_counts, master_df['subject'])


# Simple function that makes predictions from our (Naive Bayes) model

def Predict_Subject_Classification(thesis_title):
    X_new_counts = count_vect.transform([thesis_title])
    predicted = clf.predict(X_new_counts)
    return predicted[0]


