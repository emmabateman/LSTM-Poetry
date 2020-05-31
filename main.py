import tensorflow as tf
import numpy as np
import os

from poetry_scraper import PoetryScraper

reader = iter(PoetryScraper())

vocab = []
indices = {}

model = None

def vectorize(text):
    return [np.array([int(i==indices[char]) for i in range(len(vocab))]) for char in text]

def get_vocab_from_web():
    global vocab
    global indices
    global reader

    #use first 10 pages to establish character set
    first_10 = [next(reader) for i in range(10)]
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for page in first_10:
        for title, poem in page:
            chars += title
            chars += poem
    vocab = np.array(sorted(set(chars)))
    print(vocab)
    indices = {u:i for i, u in enumerate(vocab)}
    print(indices)

    #reset reader
    reader = iter(PoetryScraper())

def read_local_poems(directory):
    poems = []
    for filename in os.listdir(directory):
        f = open(directory+"/"+filename)
        title = f.readline()
        text = ""
        for line in f.readlines():
            text += line
        poems.append((title, text))

    return poems

def get_vocab_from_files(directory):
    global vocab
    global indices

    poems = read_local_poems(directory)

    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for title, poem in poems:
        chars += title
        chars += poem
    vocab = np.array(sorted(set(chars)))
    print(vocab)
    indices = {u:i for i, u in enumerate(vocab)}
    print(indices)

def init_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(vocab), 256),
        tf.keras.layers.LSTM(1024),
        tf.keras.layers.Dense(len(vocab))
        ])

    model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy)

    return model

def train_from_web():
    model = init_model()
    
    training_data = next(reader)
    while training_data:
        for title, poem in training_data:
            data = vectorize(poem)
            inputs = data[:-1]
            outputs = data[1:]
            model.fit(x=inputs, y=outputs)

        training_data = next(reader)

def train_from_files(directory):
    model = init_model()

    poems = read_local_poems(directory)
    print(poems)

    for title, poem in poems:
        data = vectorize(poem)
        inputs = np.array(data[:-1])
        outputs = np.array(data[1:])
        print(inputs.shape)
        print(outputs.shape)
        model.fit(x=inputs, y=outputs)

def generate(starter, n):
    model.reset_states()
    inputs = vectorize(starter)

    for i in range(n):
        predictions = model(inputs)
        print(predictions)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0,0]
        inputs = [int(i==predicted_id) for i in range(len(vocab))]
        starter.append(vocab[predicted_id])

        print(starter)

    return starter

def main():
    get_vocab_from_files('test_poems')
    train_from_files('test_poems')
    print("Enter first few words: ")
    print(generate(input(), 100))

main()
