import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.signal as sig

filenames = ['C:/Users/erski/Documents/NSynth/nsynth-test.tfrecord']
out_name = ['C:/Users/erski/Documents/NSynth/nsynth-test.tfrecord']
fs = 16000


# format by which single data records are parsed
def extract_fn(data_record):
    features = {
        'audio': tf.io.FixedLenFeature([64000], dtype=tf.float32),
        'pitch': tf.io.FixedLenFeature([1], dtype=tf.int64)
    }
    sample = tf.io.parse_single_example(data_record, features)
    return sample


def map_spectrogram(data_record):
    spec = tf.signal.stft(data_record['audio'], frame_length=1024, frame_step=512)
    mag = tf.abs(spec)
    return {'audio': data_record['audio'], 'spec': mag, 'pitch': data_record['pitch']}


dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(extract_fn)
dataset = dataset.shuffle(buffer_size=10000)

dataset = dataset.map(map_spectrogram)

for raw_record in dataset:
    spec = np.array(raw_record['spec'])
    print(spec.shape)

    plt.figure()
    plt.imshow(spec)
    plt.show()

    audio = np.array(raw_record['audio'])
    # f, t, Sxx = sig.spectrogram(audio, fs, nperseg=512)

    # print(np.array(raw_record['audio']), raw_record['pitch'])
    sd.play(np.array(raw_record['audio']), 16000, blocking=True)




