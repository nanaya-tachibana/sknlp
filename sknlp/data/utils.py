import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _str_feature(value):
    return _bytes_feature(value.encode('utf-8'))


def _tensor_feature(value):
    return _bytes_feature(tf.io.serialize_tensor(value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


DATA_TYPE = {
    'bytes': _bytes_feature,
    'float': _float_feature,
    'int': _int64_feature,
    'str': _str_feature,
    'tensor': _tensor_feature
}


def serialize_example(values, dtypes, names=None):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    if names is None:
        names = ['feature%d' % i for i in range(len(dtypes))]
    assert len(values) == len(dtypes) == len(names)
    feature = {
        name: DATA_TYPE[dtype](value)
        for value, dtype, name in zip(values, dtypes, names)
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    x = example_proto.SerializeToString()
    return x
