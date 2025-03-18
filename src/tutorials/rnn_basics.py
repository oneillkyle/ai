import keras
import tensorflow as tf
import tensorflow_datasets as tfds

# Vectorization function
def vectorize_text(text, label):
    return vectorize_layer(text), label

# Load data
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:80%]', 'train[80%:]', 'test'),
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
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)  # Batch the dataset

val_ds = validation_data.map(vectorize_text)  # Apply vectorization to text and labels
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)  # Batch the dataset

embedding_dim = 64
# Build the model
model = keras.Sequential([
    # vectorize_layer,  # Apply text vectorization
    keras.layers.Embedding(vocab_size, embedding_dim),
    keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=0.2)),
    keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.2)),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Logits (no activation for binary classification)
])

optimizer = keras.optimizers.Adam(learning_rate=0.000008, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Compile the model
model.compile(optimizer=optimizer,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_ds,
                    epochs=10,
                    validation_data=val_ds,
                    verbose=1)

# Display model summary
model.summary