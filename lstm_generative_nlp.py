
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from nltk import FreqDist


# In[2]:


from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, TimeDistributed, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils


# In[3]:


vocab_size=6750
max_len = 350
HIDDEN_DIM = 1000
BATCH_SIZE = 50
LAYER_NUM = 3
EPOCHS = 10
LENGTH = 15
quote_start_token = 'QUOTE_START'


# In[4]:


print('Reading data...')

filename = 'fortune_quotes.csv'
data = pd.read_csv(filename)
df = pd.DataFrame(data)
doc = df['quotes']


# In[5]:


quotes_x = []
quotes_y = []

print('Creating dictionary...')

for quote in doc:
    quote = quote.replace('/', '').replace('\n', '').replace(
        '\r', '').replace('"', '').replace('[', '').replace(
        ']', '').replace('“', '').replace('”', '').replace(
        '.', '').replace('?', '').replace(':', '').replace(
        '(', '').replace(')', '').replace(',', '').replace(
        '-', '').replace(';', '').replace('!', '')
    quote = quote.lower()
    quote = '{} {}'.format(quote_start_token, quote)
    quote = text_to_word_sequence(quote, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    quotes_x.append(np.copy(quote))
    quotes_y.append(np.copy(quote))

distribution_x = FreqDist(np.hstack(quotes_x))
quotes_vocab_x = distribution_x.most_common(vocab_size-1)

print('Formatting data for analysis...')

x_ix_to_word = [word[0] for word in quotes_vocab_x]
x_ix_to_word.insert(0, 'ZERO_TOKEN')
x_ix_to_word.append('UNKNOWN_TOKEN')

x_word_to_ix = {word:ix for ix, word in enumerate(x_ix_to_word)}

for i, quote in enumerate(quotes_x):
    for j, word in enumerate(quote):
        if word in x_word_to_ix:
            quotes_x[i][j] = x_word_to_ix[word]
        else:
            quotes_x[i][j] = x_word_to_ix['UNKNOWN_TOKEN']

distribution_y = FreqDist(np.hstack(quotes_y))
quotes_vocab_y = distribution_y.most_common(vocab_size-1)

y_ix_to_word = [word[0] for word in quotes_vocab_y]
y_ix_to_word.insert(0, 'ZERO_TOKEN')
y_ix_to_word.append('UNKNOWN_TOKEN')

y_word_to_ix = {word:ix for ix, word in enumerate(y_ix_to_word)}

for x, quote in enumerate(quotes_y):
    for y, word in enumerate(quote):
        if word in y_word_to_ix:
            quotes_y[x][y] = y_word_to_ix[word]
        else:
            quotes_y[x][y] = y_word_to_ix['UNKNOWN_TOKEN']  
            
quotes_max_len_x = max([len(quote) for quote in quotes_x])
quotes_max_len_y = max([len(quote) for quote in quotes_y])

quotes_y_data = []
for m in quotes_y:
    quotes_y_data.append(np.delete(m, 0))

quotes_x_data = pad_sequences(quotes_x, maxlen=quotes_max_len_x,
                         dtype='int64', padding='post')
quotes_y_data = pad_sequences(quotes_y_data, maxlen=quotes_max_len_y,
                              dtype='int64', padding='post')

sequences_x = np.zeros((len(quotes_x_data), quotes_max_len_x, len(x_word_to_ix)))
for i, quote in enumerate(quotes_x_data):
    for j, word in enumerate(quote):
        sequences_x[i, j, word] = 1.

sequences_y = np.zeros((len(quotes_y_data), quotes_max_len_y, len(y_word_to_ix)))
for x, quote in enumerate(quotes_y_data):
    for y, word in enumerate(quote):
        sequences_y[x, y, word] = 1.
            
print('Finished!  Parsed, cleaned and formatted {:,} quotes and {:,} unique words for analysis.'.format(
    len(quotes_x_data), len(x_word_to_ix),))


# In[ ]:


model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, len(x_word_to_ix)), return_sequences=True))
model.add(LSTM(HIDDEN_DIM, return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(len(x_word_to_ix))))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
print('\n')
print('Model Ready for Training...')
print('\n')

# def generate_text(model, LENGTH):
# 	for i in range(LENGTH):
# 		X[0, i, :][ix[-1]] = 1
# 		print(x_ix_to_word[ix[-1]], end="")
# 		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
# 		y_word.append(x_ix_to_word[ix[-1]])
# 	return ('').join(y_word)

# In[ ]:


print("Training started...")
print('\n')
for i in range(EPOCHS):
    print('INFO - Training model: Epoch: ', i+1, ' / ', EPOCHS)
    model.fit(sequences_x,
              sequences_y,
              batch_size=BATCH_SIZE,
              epochs=1,
              verbose=1,
              shuffle=False)
    ix = [2]
    y_word = [x_ix_to_word[ix[-1]]]
    X = np.zeros((1, LENGTH, len(x_word_to_ix)))
    # generate_text(model, LENGTH)
    if i % 10 == 0:
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, i))


print('Training Complete!')

