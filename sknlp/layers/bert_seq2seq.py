from __future__ import annotations
from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.seq2seq.beam_search_decoder import _beam_search_step


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertBeamSearchDecoder(tfa.seq2seq.BeamSearchDecoder):
    def __init__(
        self,
        cell: tf.keras.layers.AbstractRNNCell,
        beam_width: int,
        end_token_id: int,
        name="beam_search_decoder",
        **kwargs
    ) -> None:
        super().__init__(cell, beam_width, name=name, **kwargs)
        self._end_token = end_token_id

    def _next_inputs(
        self, inputs: list[tf.Tensor], next_token_id: tf.Tensor
    ) -> list[tf.Tensor]:
        # (batch_size,) -> (batch_size, 1)
        next_token_id = tf.expand_dims(next_token_id, 1)
        token_ids, type_ids = inputs
        mask = tf.not_equal(token_ids, 0)
        batch_size, max_sequence_length = tf.shape(token_ids)[0], tf.shape(token_ids)[1]
        sequence_length = tf.math.reduce_sum(tf.cast(mask, tf.float32), axis=-1)
        ragged_token_ids = tf.ragged.boolean_mask(
            token_ids,
            tf.sequence_mask(sequence_length - 1, maxlen=max_sequence_length),
        )
        ragged_token_ids = tf.concat(
            [
                ragged_token_ids,
                tf.cast(next_token_id, token_ids.dtype),
                tf.fill((batch_size, 1), tf.cast(self._end_token, token_ids.dtype)),
            ],
            1,
        )
        ragged_type_ids = tf.ragged.boolean_mask(type_ids, mask)
        ragged_type_ids = tf.concat(
            [
                ragged_type_ids,
                tf.ones((batch_size, 1), dtype=token_ids.dtype),
            ],
            1,
        )
        return [ragged_token_ids.to_tensor(), ragged_type_ids.to_tensor()]

    def initialize(
        self, inputs: list[tf.Tensor], state: tf.Tensor = None
    ) -> list[tf.Tensor]:
        """Initialize the decoder.
        Returns:
          `(finished, start_inputs, initial_state)`.
        Raises:
          ValueError: If `embedding` is `None` and `embedding_fn` was not set
            in the constructor.
          ValueError: If `start_tokens` is not a vector or `end_token` is not a
            scalar.
        """
        self._batch_size = tf.shape(inputs[0])[0]
        self._finished = tf.one_hot(
            tf.zeros([self._batch_size], dtype=tf.int32),
            depth=self._beam_width,
            on_value=False,
            off_value=True,
            dtype=tf.bool,
        )
        if state is None:
            state = tf.zeros((self._batch_size, self._cell.state_size))
        state = tfa.seq2seq.tile_batch(state, self._beam_width)
        self._initial_cell_state = tf.nest.map_structure(
            self._maybe_split_batch_beams, state, self._cell.state_size
        )
        dtype = tf.nest.flatten(self._initial_cell_state)[0].dtype
        log_probs = tf.one_hot(  # shape(batch_sz, beam_sz)
            tf.zeros([self._batch_size], dtype=tf.int32),
            depth=self._beam_width,
            on_value=tf.convert_to_tensor(0.0, dtype=dtype),
            off_value=tf.convert_to_tensor(float("-inf"), dtype=dtype),
            dtype=dtype,
        )
        initial_state = tfa.seq2seq.BeamSearchDecoderState(
            cell_state=self._initial_cell_state,
            log_probs=log_probs,
            finished=self._finished,
            lengths=tf.zeros([self._batch_size, self._beam_width], dtype=tf.int64),
            accumulated_attention_probs=tuple(),
        )

        inputs = tf.nest.map_structure(
            lambda x: tfa.seq2seq.tile_batch(x, self._beam_width), inputs
        )
        return [self._finished, inputs, initial_state]

    def step(
        self,
        time: tf.Tensor,
        inputs: list[tf.Tensor],
        state: tf.Tensor,
        training: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> list[tf.Tensor]:
        """Perform a decoding step.
        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          training: Python boolean. Indicates whether the layer should
              behave in training mode or in inference mode. Only relevant
              when `dropout` or `recurrent_dropout` is used.
          name: Name scope for any created operations.
        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        with tf.name_scope(name or "BeamSearchDecoderStep"):
            cell_state = state.cell_state
            cell_outputs, next_cell_state = self._cell(
                inputs, cell_state, training=training
            )
            cell_outputs = tf.nest.map_structure(
                lambda out: self._split_batch_beams(out, out.shape[1:]), cell_outputs
            )

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            beam_search_output, beam_search_state = _beam_search_step(
                time=time,
                logits=cell_outputs,
                next_cell_state=next_cell_state,
                beam_state=state,
                batch_size=self._batch_size,
                beam_width=self._beam_width,
                end_token=self._end_token,
                length_penalty_weight=self._length_penalty_weight,
                coverage_penalty_weight=self._coverage_penalty_weight,
                output_all_scores=self._output_all_scores,
            )

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_inputs = self._next_inputs(
                inputs, self._merge_batch_beams(sample_ids, s=sample_ids.shape[2:])
            )

        return [beam_search_output, beam_search_state, next_inputs, finished]

    def call(
        self,
        inputs: list[tf.Tensor],
        state: tf.Tensor,
        training: Optional[bool] = None,
        **kwargs
    ) -> tuple[
        tfa.seq2seq.FinalBeamSearchDecoderOutput,
        tfa.seq2seq.BeamSearchDecoderState,
        tf.Tensor,
    ]:
        return tfa.seq2seq.dynamic_decode(
            self,
            output_time_major=self.output_time_major,
            impute_finished=self.impute_finished,
            maximum_iterations=self.maximum_iterations,
            parallel_iterations=self.parallel_iterations,
            swap_memory=self.swap_memory,
            training=training,
            decoder_init_input=inputs,
            decoder_init_kwargs={"state": state},
        )


@tf.keras.utils.register_keras_serializable(package="sknlp")
class BertDecodeCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self, model: tf.keras.Model, vocab_size: int, name="decode_cell", **kwargs
    ) -> None:
        self.vocab_size = vocab_size
        self.bert_layer = model.get_layer(name="bert2vec")
        self.attention_mask_layer = model.get_layer("attention_mask")
        super().__init__(name=name, **kwargs)

    @property
    def state_size(self) -> int:
        return 1

    @property
    def output_size(self) -> int:
        return self.vocab_size

    def call(
        self, inputs: list[tf.Tensor], state: tf.Tensor, **kwargs
    ) -> list[tf.Tensor]:
        token_ids, type_ids = inputs
        mask = tf.not_equal(token_ids, 0)
        attention_mask = self.attention_mask_layer([type_ids, mask])
        logits_mask = tf.equal(tf.roll(type_ids, -1, 1) - tf.roll(type_ids, -2, 1), 1)
        logits = self.bert_layer([token_ids, type_ids, attention_mask, logits_mask])[-1]
        return [tf.squeeze(logits, 1), state]