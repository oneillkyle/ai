import keras
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
import string

# stopwords = ['a', 'this', 'yourselves']
# table = str.maketrans('', '', string.punctuation)

# imdb_sentences = []
# for item in train_data:
#     sentence = str(item['text'].decode('UTF-8').lower())
#     soup = BeautifulSoup(sentence)
#     sentence = soup.get_text()
#     words = sentence.split()
#     filtered_sentence = ''
#     for word in words:
#         word = word.translate(table)
#         if word not in stopwords:
#             filtered_sentence = filtered_sentence + word + ' '
#     imdb_sentences.append(filtered_sentence)

train_data = tfds.load('imdb_reviews', split='train[:80%]', as_supervised=True).batch(32).prefetch(1000)
val_data = tfds.load('imdb_reviews', split='train[80%:]').batch(32).prefetch(1000)
test_data = tfds.load('imdb_reviews', split='test').batch(32).prefetch(1000)

vectorize_layer = keras.layers.TextVectorization(
    max_tokens=100,
    standardize='lower_and_strip_punctuation',
    output_sequence_length=3
)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


max_features = 10000
sequence_length = 250
vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = train_data.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


text_batch, label_batch = next(iter(train_data))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", train_data.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))
print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = train_data.map(vectorize_text)
val_ds = val_data.map(vectorize_text)
test_ds = test_data.map(vectorize_text)
