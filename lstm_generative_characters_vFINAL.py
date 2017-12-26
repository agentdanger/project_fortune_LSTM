
# coding: utf-8

import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, TimeDistributed, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils

SEQ_LENGTH = 100
length = 250
HIDDEN_DIM = 1000
BATCH_SIZE = 50
LAYER_NUM = 3
EPOCHS = 30

print('Reading data...') 
FILENAME = 'fortune_quotes.txt' 
data = open(FILENAME, 'r', encoding='utf8').read()

print('Creating dictionary...')

data = data.replace('/', '').replace('\r', '').replace(
    '"', '').replace('[', '').replace(']', '').replace(
    '“', '').replace('”', '').replace(':', '').replace(
    '(', '').replace(')', '').replace('\x9d', '').replace(
    '\ufeff', '').replace('â', '').replace(
    '™', '').replace('œ', '').replace('¦', '').replace(
    '`', '').replace('\n', ' ')
chars = list(set(data))
VOCAB_SIZE = len(chars)

ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

print('Formatting data for analysis...')

X = np.zeros((len(data)//SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((len(data)//SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
for i in range(0, len(data)//SEQ_LENGTH):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

print('Finished!  Parsed, cleaned and formatted a document with {:,} total characters and {:,} unique characters for analysis.'.format(len(data), len(chars),))

def generate_text(model, length):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)

model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
model.add(LSTM(HIDDEN_DIM, return_sequences=True))
for i in range(LAYER_NUM - 1):
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
print('\n')
print('Model Ready for Training...')
print('\n')

print("Training started...")
print('\n')
for i in range(EPOCHS):
    print('INFO - Training model: Epoch: ', i+1, ' / ', EPOCHS)
    model.fit(X,
              y,
              batch_size=BATCH_SIZE,
              epochs=1,
              verbose=1,
              shuffle=False)
    generate_text(model, length)
    print('\n')
    if (i % 10 == 0):
        model.save_weights('checkpoint_{}_epoch_{}.hdf5'.format(HIDDEN_DIM, i))
        
print('Training Complete!')


