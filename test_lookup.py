import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow_text as text


class LookupLayer(Layer):

    def __init__(self, vocab):
        super().__init__()
        init = tf.lookup.KeyValueTensorInitializer(
            vocab,
            tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
            key_dtype=tf.string,
            value_dtype=tf.int64)
        self.lookup = tf.lookup.StaticVocabularyTable(
            init, 1, lookup_key_dtype=tf.string
        )

    def call(self, inputs):
        return self.lookup.lookup(inputs)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(None,), dtype=tf.string),
    LookupLayer(['I', 'really', 'liked', 'this', 'movie', 'not', 'my', 'favorite', '<pad>']),
])

words = tf.constant([['I', 'really', 'liked', 'this'], ['movie', 'not', 'my', '<pad>'], ['a', 'b', 'c', 'd']])
print(words)
print(model(words))

model.save("xxx", save_format="tf")

# def _CreateTable(vocab, num_oov=1):
#   init = tf.lookup.KeyValueTensorInitializer(
#       vocab,
#       tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
#       key_dtype=tf.string,
#       value_dtype=tf.int64)
#   return tf.lookup.StaticVocabularyTable(
#       init, num_oov, lookup_key_dtype=tf.string)

# reviews_data_array = ['I really liked this movie', 'not my favorite']
# reviews_labels_array = [1,0]
# train_x = tf.constant(reviews_data_array)
# train_y = tf.constant(reviews_labels_array)

# a = _CreateTable(['I', 'really', 'liked', 'this', 'movie', 'not', 'my', 'favorite'])

# def preprocess(data, labels):
#   t = text.WhitespaceTokenizer()
#   data = t.tokenize(data)
#   # data = data.merge_dims(-2,-1)
#   ids = tf.ragged.map_flat_values(a.lookup, data)
#   return (ids, labels)

# train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(2)
# train_dataset = train_dataset.map(preprocess)

# model = tf.keras.Sequential([
#   tf.keras.layers.InputLayer(input_shape=(None,), dtype='int64', ragged=True),
#   text.keras.layers.ToDense(pad_value=0, mask=True),
#   tf.keras.layers.Embedding(100, 16),
#   tf.keras.layers.LSTM(32),
#   tf.keras.layers.Dense(32, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
