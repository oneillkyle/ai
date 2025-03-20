import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Plot training and validation accuracy and loss
def plot_training_history(history):
    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:80%]', 'test'),
    as_supervised=True)

train_ds = train_data.batch(32).prefetch(tf.data.AUTOTUNE)
validation_ds = validation_data.batch(32).prefetch(tf.data.AUTOTUNE)

vocab_size = len(train_data)
sequence_length = 250
embedding_dim = 25

vectorize_layer = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(train_data.map(lambda x, y: x))  #

glove_embeddings = dict()
f = open('src/datasets/glove.twitter.27B.25d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings[word] = coefs
f.close()

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for index, word in enumerate(vectorize_layer.get_vocabulary()):
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# # Build the model
model = keras.Sequential([
    vectorize_layer,  # Apply text vectorization
    keras.layers.Embedding(vocab_size, embedding_dim, weights=[
                           embedding_matrix], trainable=False),
    keras.layers.Bidirectional(keras.layers.LSTM(
        embedding_dim, return_sequences=True, dropout=0.2)),
    keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.2)),
    keras.layers.Dense(24, activation='relu'),
    # Logits (no activation for binary classification)
    keras.layers.Dense(1, activation='sigmoid')
])

optimizer = keras.optimizers.Adam(
    learning_rate=0.000008, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile the model
model.compile(optimizer=optimizer,
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds,
                    epochs=40,
                    validation_data=validation_ds,
                    verbose=1)

# Display model summary
model.summary

plot_training_history(history)
