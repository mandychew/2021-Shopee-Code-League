import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import collections
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras. preprocessing.sequence import pad_sequences
print(tf.__version__)
from sklearn.model_selection import train_test_split

# import datasets
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

print(train.head())
print(test.head())

# regex to split raw_address string into words
my_regex = '\s|,\s|,|\.|:'

# ---------------------------------------------------------------------------------------------
# POI (Point Of Interest)

# Splitting train dataset to obtain addresses with POI
raw_df_withPOI = train[train['POI/street'].str.contains('.+(?=/)')==True]
raw_df_withPOI['POI/street'] = raw_df_withPOI['POI/street'].str.extract(r'(.+)(?=/)')

print(raw_df_withPOI.head())


# turning raw_address and POI/street columns into lists
list_of_add_withPOI = [*raw_df_withPOI['raw_address']]
list_of_POI = [*raw_df_withPOI['POI/street']]

print(list_of_add_withPOI[:5])
print(list_of_POI[:5])


# plotting distribution of length of raw_addresses and POI
from Plot_distribution import plotDistribution
plotDistribution(list_of_add_withPOI)
plotDistribution(list_of_POI)


max_len_address = 25   #len(max(raw_df_withPOI['raw_address']))
max_len_POI = 10       #len(max(raw_df_withPOI['POI/street']))


# create dictionary for short form and full words in raw_addresses and POI
dict_POI = {}


# create binary labels and adding short form words to dict_POI for all addresses with POI
from Binary_labels_and_dict import binary_labels_and_dict
labels_binary_POI, dict_POI = binary_labels_and_dict(list_of_add_withPOI, list_of_POI, dict_POI)


# filter out words that contain only digits or have 0 characters from dict_POI
from Binary_labels_and_dict import filter_dict
filtered_dict_POI = filter_dict(dict_POI)

print(len(dict_POI))
print(len(filtered_dict_POI))


# create list of sentences to tokenize
sentences_to_tokenize = list_of_add_withPOI.copy()
sentences_to_tokenize.append(list_of_POI)


# Tokenizing sentences_to_tokenize; padding list_of_add_withPOI
trunc_type = 'post'
padding = 'post'
oov_tok="<OOV>"

tokenizer_POI = Tokenizer(oov_token=oov_tok)
tokenizer_POI.fit_on_texts(sentences_to_tokenize)

word_index = tokenizer_POI.word_index
print(len(word_index))
total_words = len(word_index)+1

data_sequences = tokenizer_POI.texts_to_sequences(list_of_add_withPOI)
padded_POI = pad_sequences(data_sequences, maxlen=max_len_address, truncating=trunc_type, padding=padding)
print(padded_POI[0])
print(padded_POI.shape)


# check that tokenizer_POI works on padded_POI addresses
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded_POI[1]))
print(list_of_add_withPOI[1])


# train_test_split to get training and validation datasets with POI
train_add_withPOI, val_add_withPOI, train_labels_POI, val_labels_POI = train_test_split(
padded_POI, labels_binary_POI, test_size=0.20, random_state=42)


# define, compile and fit model_POI
embedding_dim = 64
total_words = 200000

model_POI = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, embedding_dim, input_length=max_len_address),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(max_len_address, activation='sigmoid')
])
model_POI.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_POI.summary()

earlyStopping = EarlyStopping(monitor = 'val_acc', patience = 5)
modelCheckpoint = ModelCheckpoint('best_model.hdf5', save_best_only = True)

num_epochs = 5

history = model_POI.fit(train_add_withPOI, train_labels_POI, epochs=num_epochs, validation_data=(val_add_withPOI, val_labels_POI), callbacks=[earlyStopping, modelCheckpoint])

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


# run model_POI.predict() on test dataset; convert predicted binary labels back to words (live_labels)
live_sequences = tokenizer_POI.texts_to_sequences(test['raw_address'])
live_padded = pad_sequences(live_sequences, maxlen=max_len_address, truncating=trunc_type, padding=padding)
print(live_padded.shape)

from Predict_and_convert_to_words import predict_and_convert_to_words

live_labels_POI = predict_and_convert_to_words(live_padded, test['raw_address'], 0.5, filtered_dict_POI, model_POI)
print(live_labels_POI[:5])
print(test['raw_address'][:5])


# ---------------------------------------------------------------------------------------------
# ST (Street)

# # Splitting train dataset to obtain addresses with ST
raw_df_withST = train[train['POI/street'].str.contains('(?<=/).+')==True]
raw_df_withST['POI/street'] = raw_df_withST['POI/street'].str.extract(r'(?<=/)(.+)')
raw_df_withST.head()


# turning raw_address and POI/street columns into lists
list_of_add_withST = [*raw_df_withST['raw_address']]
list_of_ST = [*raw_df_withST['POI/street']]

print(list_of_add_withST[:5])
print(list_of_ST[:5])


# plotting distribution of length of raw_addresses and ST
plotDistribution(list_of_add_withST)
plotDistribution(list_of_ST)


max_len_address = 25   #len(max(raw_df_withST['raw_address']))
max_len_ST = 10       #len(max(raw_df_withST['POI/street']))


# create dictionary for short form and full words in raw_addresses and ST
dict_ST = {}


# create binary labels and adding short form words to dict_ST for all addresses with ST
labels_binary_ST, dict_ST = binary_labels_and_dict(list_of_add_withST, list_of_ST, dict_ST)


# filter out words that contain only digits or have 0 characters from dict_ST
filtered_dict_ST = filter_dict(dict_ST)


# create list of sentences to tokenize
sentences_to_tokenize = list_of_add_withST.copy()
sentences_to_tokenize.append(list_of_ST)


# Tokenizing sentences_to_tokenize; padding list_of_add_withST
trunc_type = 'post'
padding = 'post'
oov_tok="<OOV>"

tokenizer_ST = Tokenizer(oov_token=oov_tok)
tokenizer_ST.fit_on_texts(sentences_to_tokenize)

word_index = tokenizer_ST.word_index
print(len(word_index))
total_words = len(word_index)+1

data_sequences = tokenizer_ST.texts_to_sequences(list_of_add_withST)
padded_ST = pad_sequences(data_sequences, maxlen=max_len_address, truncating=trunc_type, padding=padding)

print(padded_ST[0])
print(padded_ST.shape)


# check that tokenizer_ST works on padded_ST addresses
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded_ST[1]))
print(list_of_add_withST[1])


# train_test_split to get training and validation datasets with ST
train_add_withST, val_add_withST, train_labels_ST, val_labels_ST = train_test_split(
padded_ST, labels_binary_ST, test_size=0.20, random_state=42)


# define, compile and fit model_ST
model_ST = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, embedding_dim, input_length=max_len_address),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(max_len_address, activation='sigmoid')
])
model_ST.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_ST.summary()

num_epochs = 5

history = model_ST.fit(train_add_withST, train_labels_ST, epochs=num_epochs, validation_data=(val_add_withST, val_labels_ST), callbacks=[earlyStopping, modelCheckpoint])

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


# run model_ST.predict() on test dataset; convert predicted binary labels back to words (live_labels)
live_sequences_ST = tokenizer_ST.texts_to_sequences(test['raw_address'])
live_padded_ST = pad_sequences(live_sequences_ST, maxlen=max_len_address, truncating=trunc_type, padding=padding)
print(live_padded_ST.shape)

live_labels_ST = predict_and_convert_to_words(live_padded_ST, test['raw_address'], 0.5, filtered_dict_ST, model_ST)
print(live_labels_ST[:5])
print(test['raw_address'][:5])


# ---------------------------------------------------------------------------------------------
# Combine results of POI and ST - form 'POI/Street' column
test['POI'] = live_labels_POI
test['Street'] = live_labels_ST

print(test.head())

test['POI/Street'] = test.POI.cat(test.Street, sep='/')
print(test.head())


# Export and submit to Shopee Code League 2021 (Data Science category)
submission = test[['id', 'POI/Street']]
print(submission)

submission.to_csv('submission.csv', index=False)