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
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)  # Batch the dataset

val_ds = validation_data.map(vectorize_text)  # Apply vectorization to text and labels
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)  # Batch the dataset

# Build the model
model = keras.Sequential([
    # vectorize_layer,  # Apply text vectorization
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
model.summary