from random import Random
from networkx import project
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras_tuner import RandomSearch


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(
        lambda window: (window[:-1], window[:-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(
        hp.Int('units', min_value=128, max_value=256, step=64),
        hp.Int('kernals', min_value=3, max_value=9, step=3),
        strides=hp.Int('strides', min_value=1, max_value=3, step=1),
        padding='causal', activation='relu', input_shape=[None, 1]))
    model.add(keras.layers.Dense(28, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer=keras.optimizers.SGD(
        learning_rate=1e-5, momentum=0.5))
    return model


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, .05)
baseline = 10
amplitude = 15
slope = 0.09
noise_level = 6

# Create the series
series = baseline + trend(time, slope)
series += seasonality(time, period=365, amplitude=amplitude)
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000
dataset = windowed_dataset(
    x_train, window_size, batch_size, shuffle_buffer_size)


tuner = RandomSearch(build_model, objective='loss', max_trials=500,
                     executions_per_trial=3, directory='my_dir', project_name='cnn-tune')
tuner.search_space_summary()
tuner.search(dataset, epochs=100, verbose=2)

# model = keras.models.Sequential([
#     keras.layers.Conv1D(128, 3, strides=1,
#                         padding='causal', activation='relu', input_shape=[None, 1]),
#     keras.layers.Dense(28, activation='relu'),
#     keras.layers.Dense(10, activation='relu'),
#     keras.layers.Dense(1)
# ])

# model.compile(loss='mse', optimizer=keras.optimizers.SGD(
#     learning_rate=1e-5, momentum=0.5))

# history = model.fit(dataset, epochs=100, verbose=1)

# model.summary()

# forecast = model_forecast(model, series[..., np.newaxis], window_size)

# results = forecast[split_time - window_size:-1, -1, 0]

# mae = keras.metrics.mean_absolute_error(x_valid, results).numpy()
# print(mae)

# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid)
# plot_series(time_valid, results)
# plt.tight_layout()
# plt.show()
