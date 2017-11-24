import numpy
import spacy
import pandas as pd 
import re
#import nltk

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

nlp = spacy.load('en_core_web_sm')

vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

print('Reading data...')

filename = 'fortune_quotes.csv'
data = pd.read_csv(filename)
df = pd.DataFrame(data)
quotes = df['quotes']
quotes_parsed = nlp(quotes) 

print(quotes_parsed)

# sentences = itertools.chain(*[space.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
# # Append SENTENCE_START and SENTENCE_END
# sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentenc

# sentences = list(doc.sents)
# assert len(sentences) == 3

# def cleanText(text):
#     # get rid of newlines
#     text = text.str.strip().replace('\n', '').replace(
#     	'\r', '""').replace('"', '').replace('/', '').replace(
#     	'[', '').replace(']', '').replace('“', '').replace('”', '')
#     # replace twitter @mentions
#     mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
#     text = mentionFinder.sub("@MENTION", str(text))
#     # replace HTML symbols
#     text = text.replace("&amp;", "and").replace("&gt;", ">").replace(
#     	"&lt;","<")
#     # lowercase
#     text = text.lower()
#     return text

# def tokenizeText(sample):
#     # get the tokens using spaCy
#     tokens = parser(sample)
#     # lemmatize
#     # lemmas = []
#     # for tok in tokens:
#     #     lemmas.append(tok.lemma_.lower(
#     #     	).strip() if tok.lemma_ != "-PRON-" else tok.lower_)
#     # tokens = lemmas
#     # # stoplist the tokens
#     # tokens = [tok for tok in tokens]
#     # # stoplist symbols
#     # tokens = [tok for tok in tokens]
#     # # remove large strings of whitespace
#     # while "" in tokens:
#     #     tokens.remove("")
#     # while " " in tokens:
#     #     tokens.remove(" ")
#     # while "\n" in tokens:
#     #     tokens.remove("\n")
#     # while "\n\n" in tokens:
#     #     tokens.remove("\n\n")
#     return tokens

# clean_text = cleanText(quotes)
# tokens = tokenizeText(clean_text)

# print(clean_text)
# print(tokens)