import tensorflow as tf
import keras

import numpy as np
import os
import time

path_to_file = keras.utils.get_file(
    'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# Take a look at the first 250 characters in text
# print(text[:250])

# The unique characters in the file
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

example_texts = ['abcdefg', 'xyz']

chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
chars

ids_from_chars = keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)

ids = ids_from_chars(chars)
ids

chars_from_ids = keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
all_ids

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# for ids in ids_dataset.take(10):
#     print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# for seq in sequences.take(1):
#     print(chars_from_ids(seq))

# for seq in sequences.take(5):
#     print(text_from_ids(seq).numpy())


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

# for input_example, target_example in dataset.take(1):
#     print("Input :", text_from_ids(input_example).numpy())
#     print("Target:", text_from_ids(target_example).numpy())

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


# class MyModel(keras.Model):
#     def __init__(self, vocab_size, embedding_dim, rnn_units):
#         super().__init__()
#         self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
#         self.gru = keras.layers.GRU(rnn_units,
#                                     return_sequences=True,
#                                     return_state=True)
#         self.dense = keras.layers.Dense(vocab_size)

#     def call(self, inputs, states=None, return_state=False, training=False):
#         x = inputs
#         x = self.embedding(x, training=training)
#         if states is None:
#             # Also a fix for GPU issues
#             states = self.gru.get_initial_state(tf.shape(x)[0])
#         # x, states = self.gru(x, initial_state=states, training=training)
#         # Fix for a bug in GPU Cuda
#         r = self.gru(x, initial_state=states, training=training)
#         x, states = r[0], r[1:]
#         x = self.dense(x, training=training)

#         if return_state:
#             return x, states
#         else:
#             return x

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim),
    keras.layers.Bidirectional(keras.layers.LSTM(
        embedding_dim, return_sequences=True, dropout=0.2)),
    keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.2, return_sequences=True)),
    keras.layers.Dense(vocab_size)
])

# model = MyModel(
#     vocab_size=vocab_size,
#     embedding_dim=embedding_dim,
#     rnn_units=rnn_units)

for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print("Prediction shape:", example_batch_predictions.shape)
    print("Target shape:", target_example_batch.shape)
    print("Input shape:", input_example_batch.shape)

# sampled_indices = tf.random.categorical(
#     example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
# sampled_indices

# print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
# print()
# print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape,
#       " # (batch_size, sequence_length, vocab_size)")
# print("Mean loss:        ", example_batch_mean_loss)

# tf.exp(example_batch_mean_loss).numpy()

# Directory where the checkpoints will be saved
checkpoint_dir = 'src/training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


model.compile(optimizer='adam', loss=loss)

EPOCHS = 20

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Custom Training Option
# class CustomTraining(MyModel):
#     @tf.function
#     def train_step(self, inputs):
#         inputs, labels = inputs
#         with tf.GradientTape() as tape:
#             predictions = self(inputs, training=True)
#             loss = self.loss(labels, predictions)
#         grads = tape.gradient(loss, model.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

#         return {'loss': loss}


# model = CustomTraining(
#     vocab_size=len(ids_from_chars.get_vocabulary()),
#     embedding_dim=embedding_dim,
#     rnn_units=rnn_units)

# model.compile(optimizer=keras.optimizers.Adam(),
#               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# model.fit(dataset, epochs=1)

model.summary()


# class OneStep(keras.Model):
#     def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
#         super().__init__()
#         self.temperature = temperature
#         self.model = model
#         self.chars_from_ids = chars_from_ids
#         self.ids_from_chars = ids_from_chars

#         # Create a mask to prevent "[UNK]" from being generated.
#         skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
#         sparse_mask = tf.SparseTensor(
#             # Put a -inf at each bad index.
#             values=[-float('inf')]*len(skip_ids),
#             indices=skip_ids,
#             # Match the shape to the vocabulary
#             dense_shape=[len(ids_from_chars.get_vocabulary())])
#         self.prediction_mask = tf.sparse.to_dense(sparse_mask)

#     @tf.function
#     def generate_one_step(self, inputs, states=None):
#         # Convert strings to token IDs.
#         input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
#         input_ids = self.ids_from_chars(input_chars).to_tensor()

#         # Run the model.
#         # predicted_logits.shape is [batch, char, next_char_logits]
#         predicted_logits, states = self.model(inputs=input_ids, states=states,
#                                               return_state=True)
#         # Only use the last prediction.
#         predicted_logits = predicted_logits[:, -1, :]
#         predicted_logits = predicted_logits/self.temperature
#         # Apply the prediction mask: prevent "[UNK]" from being generated.
#         predicted_logits = predicted_logits + self.prediction_mask

#         # Sample the output logits to generate token IDs.
#         predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
#         predicted_ids = tf.squeeze(predicted_ids, axis=-1)

#         # Convert from token ids to characters
#         predicted_chars = self.chars_from_ids(predicted_ids)

#         # Return the characters and model state.
#         return predicted_chars, states


# one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# tf.saved_model.save(one_step_model, 'one_step')
# one_step_reloaded = tf.saved_model.load('one_step')

# states = None
# next_char = tf.constant(['ROMEO:'])
# result = [next_char]

# for n in range(100):
#     next_char, states = one_step_reloaded.generate_one_step(
#         next_char, states=states)
#     result.append(next_char)

# print(tf.strings.join(result)[0].numpy().decode("utf-8"))
