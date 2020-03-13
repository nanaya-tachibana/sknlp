import tensorflow as tf
from sknlp.layers import MLPLayer


def test_mlp():
    num_layers = 3
    output_size = 10
    hidden_size = 20
    mlp = MLPLayer(num_layers, hidden_size, output_size)
    assert len(mlp.dense_layers) == num_layers
    assert len(mlp.batchnorm_layers) == num_layers - 1

    batch_size = 4
    random_data = tf.random.normal((batch_size, 50))
    assert mlp(random_data).shape.as_list() == [4, output_size]
