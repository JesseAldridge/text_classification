from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction import text
# from sklearn import decomposition, ensemble

import pandas #, xgboost, numpy, textblob, string
from keras.preprocessing import text as keras_text, sequence
# from keras import layers, models, optimizers

print 'loading dataset...'
with open('corpus.txt') as f:
  data = f.read()
labels, texts = [], []
for i, line in enumerate(data.splitlines()):
  content = line.split(' ', 1)
  labels.append(content[0])
  texts.append(content[1])

print 'creating dataframe...'
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

print 'splitting into training and validation sets...'
train_texts, valid_texts, train_labels, valid_labels = model_selection.train_test_split(
  trainDF['text'], trainDF['label']
)

print 'encoding labels...'
encoder = preprocessing.LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
valid_labels = encoder.fit_transform(valid_labels)

print 'count vectorizing texts...'
count_vect = text.CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])
text_train_count =  count_vect.transform(train_texts)
text_valid_count =  count_vect.transform(valid_texts)

print 'tf-idf word-level vectorizing texts...'
tfidf_vect = text.TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
text_train_tfidf =  tfidf_vect.transform(train_texts)
text_valid_tfidf =  tfidf_vect.transform(valid_texts)

print 'tf-idf ngram-level vectorizing texts...'
tfidf_vect_ngram = text.TfidfVectorizer(
  analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000
)
tfidf_vect_ngram.fit(trainDF['text'])
text_train_tfidf_ngram =  tfidf_vect_ngram.transform(train_texts)
text_valid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_texts)

print 'tf-idf characer level vectorizing texts...'
tfidf_vect_ngram_chars = text.TfidfVectorizer(
  analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000
)
tfidf_vect_ngram_chars.fit(trainDF['text'])
text_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_texts)
text_valid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_texts)

print 'loading pre-trained word-embedding vectors...'
embeddings_index = {}
for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

print 'tokenizing texts...'
token = keras_text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_texts = sequence.pad_sequences(token.texts_to_sequences(train_texts), maxlen=70)
valid_seq_texts = sequence.pad_sequences(token.texts_to_sequences(valid_texts), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


import pdb; pdb.set_trace()
