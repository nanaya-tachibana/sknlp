# -*- coding: utf-8 -*-
import json
import os
import shutil
import tempfile

import tensorflow as tf
from tensorflow.keras.layers import Embedding

from sknlp.vocab import Vocab
from sknlp.utils import make_tarball


class Token2vec:

    def __init__(self, vocab, embed_size, segmenter='list',
                 name='token2vec', **kwargs):
        """
        基础符号->向量模块.

        Parameters:
        ----------
        vocab: sknlp.vocab.Vocab. 符号表.
        embed_size: int > 0. 向量维度.
        name: str. 模块名.
        embeddings_initializer: Initializer for the `embeddings` matrix.
        embeddings_regularizer: Regularizer function applied to
        the `embeddings` matrix.
        embeddings_constraint: Constraint function applied to
        the `embeddings` matrix.
        input_length: Length of input sequences, when it is constant.
        This argument is required if you are going to connect
        `Flatten` then `Dense` layers upstream
        (without it, the shape of the dense outputs cannot be computed).

        Inputs
        ----------
        2D tensor with shape: `(batch_size, input_length)`.

        Outputs
        ----------
        3D tensor with shape: `(batch_size, input_length, embed_size)`.
        """
        self._vocab = vocab
        self._embedding = Embedding(len(vocab), embed_size, mask_zero=True,
                                    name='embeddings', **kwargs)
        self._model = tf.keras.Sequential(self._embedding, name=name)
        self._embed_size = embed_size
        self._segmenter = segmenter
        self._kwargs = kwargs

    def __call__(self, inputs):
        return self._embedding(inputs)

    def compute_mask(self, inputs):
        return self._embedding.compute_mask(inputs)

    def save(self, filename):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._model.save_weights(os.path.join(temp_dir, 'weights.h5'))
            with open(os.path.join(temp_dir, 'vocab.json'), 'w') as f:
                f.write(self._vocab.to_json())
            with open(os.path.join(temp_dir, 'meta.json'), 'w') as f:
                f.write(json.dumps(self.get_config()))

            if not filename.endswith('.tar'):
                filename = '.'.join([filename, 'tar'])
            make_tarball(filename, temp_dir)

    @classmethod
    def load(cls, filename):
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.unpack_archive(filename, temp_dir, format='tar')
            with open(os.path.join(temp_dir, 'vocab.json')) as f:
                vocab = Vocab.from_json(f.read())
            with open(os.path.join(temp_dir, 'meta.json')) as f:
                meta = json.loads(f.read())

            module = cls(vocab, **meta)
            module._model.load_weights(os.path.join(temp_dir, 'weights.h5'))
            return module

    def freeze(self):
        for layer in self._model.layers:
            layer.trainable = False

    def get_config(self):
        return {'embed_size': self._embed_size,
                'segmenter': self._segmenter,
                **self._kwargs}

    @property
    def vocab(self):
        return self._vocab

    @property
    def segmenter(self):
        return self._segmenter

    @property
    def embed_size(self):
        return self._embed_size

    @property
    def input(self):
        return self._model.input

    @property
    def output(self):
        return self._model.output

    @property
    def input_shape(self):
        return self._model.input.shape

    @property
    def output_shape(self):
        return self._model.output.shape

    @property
    def weights(self):
        return self._model.get_weights()

    def set_weights(self, weights):
        return self._model.set_weights(weights)
