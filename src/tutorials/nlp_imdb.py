import string
import re
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from bs4 import BeautifulSoup

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

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')




# vectorize_layer = keras.layers.TextVectorization(
#     max_tokens=100,
#     standardize='lower_and_strip_punctuation',
#     output_sequence_length=3
# )




# def vectorize_text(text, label):
#     text = tf.expand_dims(text, -1)
#     return vectorize_layer(text), label


# text_batch, label_batch = next(iter(train_data))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", train_data.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))
# print("1287 ---> ", vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ", vectorize_layer.get_vocabulary()[313])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

# train_ds = train_data.map(vectorize_text)
# val_ds = val_data.map(vectorize_text)
# test_ds = test_data.map(vectorize_text)

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
# hub_layer = hub.KerasLayer(embedding, input_shape=[],
#                            dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])
# hub_layer_wrapper = keras.layers.Lambda(lambda x: hub_layer(x))

vocab_size = 6000
sequence_length = 250
vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = train_data.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

model = keras.Sequential()
model.add(vectorize_layer)
model.add(keras.layers.Embedding(6000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data,
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

model.summary()

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
