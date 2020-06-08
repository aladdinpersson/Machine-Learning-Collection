# -*- coding: utf-8 -*-

"""
Having created our vocabulary we now need to create
the dataset X,y which we will create by doing frequency
vector for each email. For example if our vocabulary
has the words

[aardkvark, ..., buy, ... money, .... zulu]

We go through each email and count up how many times each
word was repeated, so for a specific example this might look
like:
    
[0, ..., 4, ... 2, .... 0] 

And perhaps since both "buy" and "money" this email might be
spam

"""
import pandas as pd
import numpy as np
import ast

data = pd.read_csv("data/emails.csv")
file = open("vocabulary.txt", "r")
contents = file.read()
vocabulary = ast.literal_eval(contents)

X = np.zeros((data.shape[0], len(vocabulary)))
y = np.zeros((data.shape[0]))

for i in range(data.shape[0]):
    email = data.iloc[i, :][0].split()

    for email_word in email:
        if email_word.lower() in vocabulary:
            X[i, vocabulary[email_word]] += 1

    y[i] = data.iloc[i, :][1]

# Save stored numpy arrays
np.save("data/X.npy", X)
np.save("data/y.npy", y)
