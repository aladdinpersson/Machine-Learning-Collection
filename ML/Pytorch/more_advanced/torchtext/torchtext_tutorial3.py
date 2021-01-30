import spacy
import pandas as pd
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split

### Load data from two text files where each row is a sentence ###
english_txt = open("train_WMT_english.txt", encoding="utf8").read().split("\n")
german_txt = open("train_WMT_german.txt", encoding="utf8").read().split("\n")

raw_data = {
    "English": [line for line in english_txt[1:100]],
    "German": [line for line in german_txt[1:100]],
}

df = pd.DataFrame(raw_data, columns=["English", "German"])

# create train and test set
train, test = train_test_split(df, test_size=0.1)

# Get train, test data to json and csv format which can be read by torchtext
train.to_json("train.json", orient="records", lines=True)
test.to_json("test.json", orient="records", lines=True)

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

### Now we're back to where we were in previous Tutorials ###

"""
To install spacy languages use:
python -m spacy download en
python -m spacy download de
"""

spacy_eng = spacy.load("en")
spacy_ger = spacy.load("de")


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True, tokenize=tokenize_ger, lower=True)

fields = {"English": ("eng", english), "German": ("ger", german)}

train_data, test_data = TabularDataset.splits(
    path="", train="train.json", test="test.json", format="json", fields=fields
)

english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), batch_size=32, device="cuda"
)

for batch in train_iterator:
    print(batch)
