import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
plt.style.use('bmh')
import seaborn as sns
import nltk
from nltk import word_tokenize, WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

import os
import collections

from gensim import models 
import gensim
from gensim.models import Word2Vec

#For Building Model 
import tensorflow as tf 
import keras 
from keras import regularizers, backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Embedding, BatchNormalization, LSTM, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from mlxtend.plotting import plot_confusion_matrix


#load training data set
path='C:\\Users\\15366\OneDrive\\桌面\\Sentiment-Analysis-using-CNN-and-Word2Vec\\stanfordSentimentTreebank\\train.csv'
df=pd.read_csv(path)
original_sentence=df['sentences_original']
sen_values = df ['sen_values']
label=df['label']
lemmatized_tokens = df ['lemmatized_tokens']

tokens = [word_tokenize(sen) for sen in df['sentences_original']]
df['tokens'] = tokens
data_train, data_test = train_test_split(df, test_size=0.2, random_state=42)
print("Data train : %s, Data test:  %s." % (data_train.shape[0], data_test.shape[0]))

# data train
all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))

print("Data train total words : %s , Vocabulary size : %s" % (len(all_training_words), len(TRAINING_VOCAB)))




#data test
all_test_words = [word for tokens in data_test["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("Data test total words : %s , Vocabulary size : %s" % (len(all_test_words), len(TEST_VOCAB)))


# word2vec = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\15366\\OneDrive\\桌面\\Sentiment-Analysis-using-CNN-and-Word2Vec\\GoogleNews-vectors-negative300.bin.gz', 
#                                                            binary=True)
data_train['doc_len'] = data_train['sentences_original'].apply(lambda words: len(words.split(' ')))
max_seq_len = np.max(data_train['doc_len'])+1


print(data_train['doc_len'])
#Plot 
fig, ax = plt.subplots(figsize=(6,4))

data_train['doc_len'].plot(kind='hist',
                      density=True,
                      alpha=0.65,
                      bins=15,
                      label="Tweet's Frequency")

data_train['doc_len'].plot(kind='kde', label='')

ax.set_xlim(-5, 22)
ax.set_xlabel("")
ax.set_ylim(0, 0.13)
ax.set_yticks([])
ax.set_ylabel("")
ax.set_title("Word Distribution per Tweet")
ax.grid(False)
ax.axvline(x=max_seq_len, alpha=0.65, color='k', linestyle=':', label='Max Sequence lenth')
ax.tick_params(left = False, bottom = False)
for ax, spine in ax.spines.items():
    spine.set_visible(False)

plt.legend()
plt.show()

raw_docs_train = data_train['sentences_original'].tolist()
raw_docs_test  = data_test['sentences_original'].tolist()



#Training Vocab
# MAX_NB_WORDS = 100000 
#Tokenizing input data
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB),
                      lower=True,
                      char_level=False)
tokenizer.fit_on_texts(raw_docs_train)
word_seq_train = tokenizer.texts_to_sequences(raw_docs_train)
word_seq_test  = tokenizer.texts_to_sequences(raw_docs_test)

word_index = tokenizer.word_index
print('Dictionary Size: ', len(word_index))