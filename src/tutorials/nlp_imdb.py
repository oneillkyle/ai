import keras
import tensorflow as tf
import tensorflow_datasets as tfds

# Vectorization function
def vectorize_text(text, label):
    return vectorize_layer(text), label

# Load data
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:10%]', 'train[10%:20%]', 'test'),
    as_supervised=True)

# Define parameters
vocab_size = 2500
sequence_length = 250

# Vectorization layer
vectorize_layer = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Extract text data (only text, not labels) for vectorization
train_text = train_data.map(lambda x, y: x)  # Only text data
vectorize_layer.adapt(train_text)  # Adapt vectorizer to training text data

# Apply vectorization function to datasets and ensure batching
train_ds = train_data.map(vectorize_text)  # Apply vectorization to text and labels
train_ds = train_ds.batch(32)  # Batch the dataset

val_ds = validation_data.map(vectorize_text)  # Apply vectorization to text and labels
val_ds = val_ds.batch(32)  # Batch the dataset

# Build the model
model = keras.Sequential([
    vectorize_layer,  # Apply text vectorization
    keras.layers.Embedding(vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Logits (no activation for binary classification)
])

# Compile the model
model.compile(optimizer='adam',
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds,
                    epochs=10,
                    validation_data=val_ds,
                    verbose=1)

# Display model summary
model.summary()



# def custom_standardization(input_data):
#     lowercase = tf.strings.lower(input_data)
#     stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
#     return tf.strings.regex_replace(stripped_html,
#                                     '[%s]' % re.escape(string.punctuation),
#                                     '')


# vectorize_layer = keras.layers.TextVectorization(
#     max_tokens=100,
#     standardize='lower_and_strip_punctuation',
#     output_sequence_length=3
# )

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

# model.add(hub_layer)
# model.add(vectorize_layer)

# train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))

# embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
# hub_layer = hub.KerasLayer(embedding, input_shape=[],
#                            dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])
# hub_layer_wrapper = keras.layers.Lambda(lambda x: hub_layer(x))

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

# results = model.evaluate(test_data.batch(512), verbose=2)
