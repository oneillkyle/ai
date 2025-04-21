import keras
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

version_fn = getattr(keras, "version", None)
if version_fn and version_fn().startswith("3."):
    import tf_keras as keras
else:
    import keras


(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,)  # type: ignore


def format_image(image, label):
    IMAGE_SIZE = [224, 224]  # should get this dynamically
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0  # type: ignore
    return image, label


num_examples = metadata.splits['train'].num_examples
num_classes = metadata.features['label'].num_classes  # type: ignore
print(num_examples)
print(num_classes)

BATCH_SIZE = 32
train_batches = raw_train.shuffle(
    num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = raw_validation.map(
    format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = raw_test.map(format_image).batch(1)

module_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, FV_SIZE = module_selection

MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(
    handle_base)
IMAGE_SIZE = (pixels, pixels)
feature_extractor = hub.KerasLayer(
    MODULE_HANDLE, input_shape=IMAGE_SIZE + (3,), output_shape=[FV_SIZE], trainable=False)

model = keras.Sequential([
    feature_extractor,
    keras.layers.Dense(num_classes, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_batches, epochs=5, validation_data=validation_batches)

CATS_VS_DOGS_SAVED_MODEL = "src/models/cats_dogs_model"
tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)
tflite_model = converter.convert()
tflite_model_file = '{}/converted_model.tflite'.format(
    CATS_VS_DOGS_SAVED_MODEL)

with open(tflite_model_file, 'wb') as f:
    f.write(tflite_model)
