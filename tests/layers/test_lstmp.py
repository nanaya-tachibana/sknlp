import tensorflow as tf
from sknlp.layers import LSTMPCell, LSTMP

units = 32
projection_size = 8
dropout = 0.5
recurrent_dropout = 0.5
recurrent_clip = 3
projection_clip = 3
batch_size = 4
sequence_length = 5


def test_ltmpcell():
    cellv1 = LSTMPCell(units, projection_size,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       recurrent_clip=recurrent_clip,
                       projection_clip=projection_clip,
                       implementation=1)
    cellv2 = LSTMPCell(units, projection_size,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       recurrent_clip=recurrent_clip,
                       projection_clip=projection_clip,
                       implementation=2)

    random_data = tf.random.normal((batch_size, 50))
    initial_states = cellv1.get_initial_state(batch_size=batch_size,
                                              dtype=tf.float32)
    h, (_, c) = cellv1(random_data, initial_states)
    assert h.shape.as_list() == [batch_size, projection_size]
    assert c.shape.as_list() == [batch_size, units]

    h, (_, c) = cellv2(random_data, initial_states)
    assert h.shape.as_list() == [batch_size, projection_size]
    assert c.shape.as_list() == [batch_size, units]


def test_lstmcell_config():
    cell = LSTMPCell(units, projection_size)
    assert LSTMPCell.from_config(cell.get_config()).units == cell.units


def test_lstmp():
    random_data = tf.random.normal((batch_size, sequence_length, 50))
    rnn = LSTMP(units, projection_size)
    assert rnn(random_data).shape.as_list() == [batch_size, projection_size]


def test_lstmp_config():
    rnn = LSTMP(units, projection_size)
    assert LSTMP.from_config(rnn.get_config()).units == rnn.units
