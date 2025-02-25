import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

model = keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    keras.layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(300, 300, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    keras.layers.Flatten(),
    # 512 neuron hidden layer
    keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where
    # 0 for 1 class ('horses') and 1 for the other ('humans')
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    './datasets/horse-or-human/training/',
    target_size=(300, 300),
    class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(
    './datasets/horse-or-human/validation/',
    target_size=(300, 300),
    class_mode='binary')

history = model.fit(train_generator, epochs=15,
                    validation_data=validation_generator)

# model.evaluate(test_images, test_labels)

# classifications = model.predict(test_images)
# print(classifications[0])
# print(test_labels[0])
model.summary()
