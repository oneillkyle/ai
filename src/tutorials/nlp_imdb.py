import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import string

stopwords = ['a', 'this', 'yourselves']
table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ''
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + ' '
    imdb_sentences.append(filtered_sentence)

vectorize_layer = keras.layers.TextVectorization(
    max_tokens=100,
    standardize='lower_and_strip_punctuation',
    output_sequence_length=3
)

vectorize_layer.adapt(imdb_sentences)

vectorize_layer.get_vocabulary()

sentences_to_tokens = vectorize_layer([
    'i love tensorflow',
    'i love playing with this'
])

string_lookup = keras.layers.StringLookup(
    vocabulary=vectorize_layer.get_vocabulary(include_special_tokens=False), invert=True)
print(string_lookup(sentences_to_tokens - 1))
