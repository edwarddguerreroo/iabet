import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Union, Callable, List, Tuple, Optional

class IndRNN(Layer):
    def __init__(self, units: int,
                 recurrent_initializer: str = "orthogonal",
                 activation: Union[str, Callable] = "relu",
                 **kwargs):
        super(IndRNN, self).__init__(**kwargs)
        self.units = units
        self.recurrent_initializer = recurrent_initializer
        self.activation = tf.keras.activations.get(activation)
        self.state_size = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units,),
            initializer=self.recurrent_initializer,
            name="recurrent_kernel",
            constraint=tf.keras.constraints.MaxNorm(
                max_value=10.0, axis=0  # |u_j| <= c
            )
        )
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")
        self.built = True

    def call(self, inputs: tf.Tensor, states: List[tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        prev_output = states[0]
        output = tf.matmul(inputs, self.kernel) + prev_output * self.recurrent_kernel + self.bias
        output = self.activation(output)  # Aplicar la activación
        return output, [output]

    def get_initial_state(self, inputs: Optional[tf.Tensor] = None, batch_size: Optional[int] = None, dtype: Optional[tf.DType] = None) -> List[tf.Tensor]:
        return [tf.zeros((batch_size, self.units), dtype=dtype)]

    def get_config(self):
        config = super(IndRNN, self).get_config()
        config.update({
            'units': self.units,
            'recurrent_initializer': self.recurrent_initializer,
            'activation': self.activation
        })
        return config