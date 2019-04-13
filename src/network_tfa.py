import tensorflow
from tensorflow._api.v1.keras.layers import *
from tensorflow._api.v1.keras.models import Sequential


class Network(object):

    def __init__(self, properties, embed_data):
        self.properties = properties
        self.embed_data = embed_data

    def build_model(self):
        word_input = Sequential()
        word_input.add(Embedding(input_dim=self.embed_data.num_words,
                                 output_dim=self.properties.word_embed_dim,
                                 input_length=20))

        pos_input = Sequential()
        pos_input.add(Embedding(input_dim=self.embed_data.num_pos,
                                output_dim=self.properties.pos_embed_dim,
                                input_length=20))

        label_input = Sequential()
        label_input.add(Embedding(input_dim=self.embed_data.num_labels,
                                  output_dim=self.properties.label_embed_dim,
                                  input_length=12))

        dense_layers = Sequential()
        dense_layers.add(Concatenate([word_input, pos_input, label_input]))

        dense_layers.add(Dense(10))
        dense_layers.add(Activation('relu'))

        dense_layers.add(Dense(self.embed_data.num_actions()))
        dense_layers.add(Activation('softmax'))

        dense_layers.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Convert labels to categorical one-hot encoding
        one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

        # Train the model, iterating on the data in batches of 32 samples
        model.fit(data, one_hot_labels, epochs=10, batch_size=32)







