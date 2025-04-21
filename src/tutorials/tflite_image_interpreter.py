import keras
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

from utils.plotting import plot_image

CATS_VS_DOGS_SAVED_MODEL = "src/models/cats_dogs_model"

tflite_model_file = '{}/converted_model.tflite'.format(
    CATS_VS_DOGS_SAVED_MODEL)

interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
ouput_index = interpreter.get_output_details()[0]['index']

predictions = []

def format_image(image, label):
    IMAGE_SIZE = [224, 224]  # should get this dynamically
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0  # type: ignore
    return image, label

raw_test, metadata = tfds.load(
    'cats_vs_dogs',
    split='train[80%:]',
    with_info=True,
    as_supervised=True,)  # type: ignore

test_batches = raw_test.map(format_image).batch(1) # type: ignore

test_labels, test_imgs = [], []
for img, label in test_batches.take(100):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(ouput_index))
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)
    
score = 0
for item in range(0, 99):
    prediction = np.argmax(predictions[item])
    label = test_labels[item]
    if prediction == label:
        score = score + 1

print("Out of 100 predictions I got {} correct".format(str(score)))

# for index in range(0, 99):
#     plt.figure(figsize=(6, 3))
#     plt.subplot(1, 2, 1)
#     plot_image(index, predictions, test_labels, test_imgs, ['cat', 'dog'])
# plt.show()