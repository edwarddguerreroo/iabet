import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, MaxPool1D, Dropout, Dense, LayerNormalization, Embedding, Add, Reshape, \
    TimeDistributed, MultiHeadAttention, Attention, GlobalAveragePooling1D, Concatenate
import numpy as np
from typing import Optional, Tuple, List, Union, Callable
from tensorflow.keras import activations


# --- Gated Linear Unit (GLU) ---
class GLU(Layer):
    def __init__(self, units: Optional[int] = None, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1] // 2  # Asignar la mitad si no se especifica
        self.dense = tf.keras.layers.Dense(self.units * 2)  # Duplicamos para hacer el split
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.dense(inputs)  # Aplicamos transformación lineal
        a, b = tf.split(x, 2, axis=-1)  # Partimos en dos
        return a * tf.sigmoid(b)  # Aplicamos activación sigmoide

    def get_config(self):
        config = super(GLU, self).get_config()
        config.update({"units": self.units})
        return config


# --- Gated Residual Network (GRN) ---

class GatedResidualNetwork(Layer):
    def __init__(self, units, dropout_rate, activation="elu", 
                 use_time_distributed=False, use_glu=True, use_layer_norm=True,
                 l1_reg=0.0, l2_reg=0.0, kernel_initializer="glorot_uniform", **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.activation = activation
        self.use_glu = use_glu
        self.use_layer_norm = use_layer_norm
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.kernel_initializer = kernel_initializer
        
        if use_layer_norm:
            self.layer_norm = LayerNormalization()
        
        self.dense1 = Dense(units, activation=self.activation,
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                            kernel_initializer=self.kernel_initializer)
        
        if use_glu:
            self.glu = Dense(units, activation="linear",
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                             kernel_initializer=self.kernel_initializer)
        else:
            self.dense2 = Dense(units, activation="linear",
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg),
                                kernel_initializer=self.kernel_initializer)
        
        self.dropout = Dropout(dropout_rate)
        self.gate = Dense(units, activation="sigmoid")

        # Función para aplicar TimeDistributed si use_time_distributed=True
        self.apply_time_distributed = lambda layer: TimeDistributed(layer) if use_time_distributed else layer

        # Proyección de inputs antes de la conexión residual
        self.residual_projection = Dense(units, kernel_initializer=self.kernel_initializer)

    def call(self, inputs, context=None, training=None):
        if self.use_layer_norm:
            x = self.layer_norm(inputs)
        else:
            x = inputs

        if x.shape[-1] != self.units:
            x = self.apply_time_distributed(self.residual_projection)(x)

        if context is not None:
            if len(x.shape) == 3 and len(context.shape) == 2:
                context = tf.expand_dims(context, axis=1)
                context = tf.tile(context, [1, tf.shape(x)[1], 1])
            x = Concatenate(axis=-1)([x, context])

        x = self.apply_time_distributed(self.dense1)(x)
        x = self.dropout(x, training=training)

        if self.use_glu:
            x = self.apply_time_distributed(self.glu)(x)
        else:
            x = self.apply_time_distributed(self.dense2)(x)

        gate = self.apply_time_distributed(self.gate)(x)
        x = x * gate
        x = self.dropout(x, training=training)

        # Proyectar inputs si es necesario antes de sumarlo
        residual = inputs
        if residual.shape[-1] != x.shape[-1]:
            residual = self.apply_time_distributed(self.residual_projection)(residual)
        
        assert residual.shape == x.shape, f"Incompatible shapes: {residual.shape} vs {x.shape}"
        
        return residual + x  # Residual connection

    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate,
            "use_time_distributed": self.use_time_distributed,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "use_glu": self.use_glu,
            "use_layer_norm": self.use_layer_norm,
            "l1_reg": self.l1_reg,
            "l2_reg": self.l2_reg
        })
        return config


# --- Variable Selection Network (VSN) ---
class VariableSelectionNetwork(Layer):
    def __init__(self, num_inputs: int, units: int, dropout_rate: float,
                 use_glu_in_grn: bool = True, l1_reg: float = 0.0,
                 l2_reg: float = 0.0, context_units: Optional[int] = None, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.context_units = context_units
        self.num_inputs = num_inputs
        self.units = units  # Asegurar que units = 16
        self.dropout_rate = dropout_rate
        self.use_glu_in_grn = use_glu_in_grn
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        # Regularización
        regularizer = tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)

        # GRN para cada entrada (transforma de 8 a 16)
        self.grns = [
            GatedResidualNetwork(units, dropout_rate, activation="elu", use_glu=use_glu_in_grn,
                                 l1_reg=l1_reg, l2_reg=l2_reg)
            for _ in range(num_inputs)
        ]

        # GRN de agregación (transforma a units=16)
        self.grn_agg = GatedResidualNetwork(units=units, dropout_rate=dropout_rate, activation="elu",
                                            use_glu=use_glu_in_grn, l1_reg=l1_reg, l2_reg=l2_reg)

        # Pesos de selección de variables
        self.softmax = Dense(num_inputs, activation="softmax", use_bias=False, kernel_regularizer=regularizer)

    def call(self, inputs: List[tf.Tensor], training=None, context: Optional[tf.Tensor] = None):
        if not isinstance(inputs, list) or len(inputs) != self.num_inputs:
            raise ValueError(f"Expected list of {self.num_inputs} tensors, but got {type(inputs)} with length {len(inputs)}")

        # Aplicar GRN a cada entrada (transforma de 8 a 16)
        grn_outputs = [self.grns[i](inp, training=training, context=context) for i, inp in enumerate(inputs)]

        # Concatenar las salidas de los GRNs
        grn_outputs_concat = Concatenate(axis=-1)(grn_outputs)  # (batch_size, seq_len, num_inputs * units)

        # Aplicar GRN de agregación (transforma a units=16)
        grn_aggregate = self.grn_agg(grn_outputs_concat, training=training, context=context)

        # Reducir la dimensionalidad de grn_aggregate a (batch_size, num_inputs)
        grn_aggregate_reduced = tf.reduce_mean(grn_aggregate, axis=1)  # (batch_size, units)

        # Cálculo de pesos de selección
        sparse_weights = self.softmax(grn_aggregate_reduced)  # (batch_size, num_inputs)

        # Expandir dimensiones de sparse_weights para que coincidan con las entradas
        sparse_weights_expanded = tf.expand_dims(sparse_weights, axis=1)  # (batch_size, 1, num_inputs)
        sparse_weights_expanded = tf.expand_dims(sparse_weights_expanded, axis=-1)  # (batch_size, 1, num_inputs, 1)

        # Ponderar entradas asegurando broadcast correcto
        weighted_inputs = [grn_outputs[i] * sparse_weights_expanded[:, :, i, :] for i in range(self.num_inputs)]

        # Sumar entradas ponderadas para obtener la salida final
        outputs = Add()(weighted_inputs)  # (batch_size, seq_len, units)
        return outputs, sparse_weights

# --- Positional Encoding ---
class PositionalEmbedding(Layer):
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        # Crear la matriz de positional encoding
        pe = tf.Variable(self._get_positional_encoding(max_len, d_model), trainable=False)
        self.pe = tf.expand_dims(pe, 0)  # [1, max_len, d_model]

    def _get_positional_encoding(self, max_len: int, d_model: int) -> tf.Tensor:
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pe = tf.zeros((max_len, d_model))
        pe_sin = tf.sin(positions * div_term)
        pe_cos = tf.cos(positions * div_term)

        pe_final = tf.concat([
            tf.expand_dims(pe_sin, axis=2),
            tf.expand_dims(pe_cos, axis=2),
        ], axis=2)
        return tf.reshape(pe_final, (max_len, d_model))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]  # Sumar el positional encoding

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'max_len': self.max_len,
        })
        return config


# --- Time2Vec ---
class Time2Vec(Layer):
    def __init__(self, output_dim: int, kernel_initializer: str = "glorot_uniform", activation: str = 'sin', **kwargs):
        super(Time2Vec, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.kernel_initializer = kernel_initializer
        self.activation = activation  # Funcion de activacion

        if self.activation == 'sin':
            self.act_func = tf.sin
        elif self.activation == 'cos':
            self.act_func = tf.cos
        else:
            self.act_func = tf.keras.activations.get(activation)  # Otra funcion

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True,
                                 name="W")
        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer="zeros",
                                 trainable=True,
                                 name="b")
        # Parametros para el periodico
        self.W_p = self.add_weight(shape=(input_shape[-1], self.output_dim),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name="W_periodic")

        self.b_p = self.add_weight(shape=(self.output_dim,),
                                   initializer="zeros",
                                   trainable=True,
                                   name="b_periodic")
        self.built = True

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, seq_len, 1)  (asumiendo que la entrada es el tiempo)
        original = tf.matmul(x, self.W) + self.b  # Parte lineal
        periodic = self.act_func(tf.matmul(x, self.W_p) + self.b_p)  # Parte periódica
        return tf.concat([original, periodic], axis=-1)  # Concatenar

    def get_config(self):
        config = super(Time2Vec, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'kernel_initializer': self.kernel_initializer,
            'activation': self.activation
        })
        return config


# --- Learnable Fourier Features ---
class LearnableFourierFeatures(Layer):
    def __init__(self, num_features: int, output_dim: int, **kwargs):
        super(LearnableFourierFeatures, self).__init__(**kwargs)
        self.num_features = num_features
        self.output_dim = output_dim
        # Inicializar frecuencias y amplitudes aleatoriamente
        self.freqs = self.add_weight(shape=(num_features, output_dim),
                                    initializer="random_uniform",
                                    trainable=True,
                                    name="frequencies")
        self.amps = self.add_weight(shape=(num_features, output_dim),
                                    initializer="random_normal",
                                    trainable=True,
                                    name="amplitudes")
        # Podriamos agregar fase, pero por ahora lo dejaremos asi
        # self.phases = self.add_weight(shape=(num_features, output_dim),
        #                             initializer="random_uniform",
        #                             trainable=True,
        #                             name="phases")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, seq_len, 1)  (tiempo)
        projected = 2 * np.pi * tf.matmul(x, self.freqs)  # (batch_size, seq_len, output_dim)
        # Usamos la formula: A * cos(2πft + Φ)
        return self.amps * tf.cos(projected)  # + self.phases #Podriamos agregar una fase

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'output_dim': self.output_dim
        })
        return config


# --- Sparsemax ---
class Sparsemax(Layer):
    """Sparsemax activation function. (https://arxiv.org/abs/1602.02068)"""

    def __init__(self, axis: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # Verificar la forma de la máscara y los logits
        tf.print("Shape of inputs:", tf.shape(inputs))
        if mask is not None:
            tf.print("Shape of mask:", tf.shape(mask))
            tf.debugging.assert_shapes([(inputs, mask.shape)], message="Mask and inputs must have compatible shapes")
        
        # Evita valores grandes restando el máximo
        inputs_safe = inputs - tf.reduce_max(inputs, axis=self.axis, keepdims=True)
        
        # Ordenar valores en orden descendente
        z_sorted = tf.sort(inputs_safe, axis=self.axis, direction='DESCENDING')
        
        # Cálculo de tau
        range_z = tf.cast(tf.range(1, tf.shape(z_sorted)[self.axis] + 1), dtype=inputs.dtype)
        bound = 1 + range_z * z_sorted
        cumsum = tf.cumsum(z_sorted, axis=self.axis)
        is_gt = tf.cast(tf.greater(bound, cumsum), dtype=inputs.dtype)
        k = tf.reduce_max(is_gt * range_z, axis=self.axis, keepdims=True)
        tau = (tf.reduce_sum(z_sorted * is_gt, axis=self.axis, keepdims=True) - 1) / tf.maximum(k, 1.0)
        output = tf.maximum(tf.zeros_like(inputs), inputs_safe - tau)
        
        # Aplicar máscara si está presente
        if mask is not None:
            mask = tf.cast(mask, dtype=tf.bool)  # Asegurar que la máscara sea booleana
            mask = tf.broadcast_to(mask, tf.shape(output))  # Asegurar la misma forma
            output = tf.where(mask, output, tf.zeros_like(output))  # Enmascarar valores
        
        # Penalizar valores enmascarados antes de aplicar Sparsemax
        if mask is not None:
            inputs = tf.where(mask, inputs, -1e9)
            output = self.call(inputs)  # Reaplicar la función con valores penalizados
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

# --- DropConnect ---
class DropConnect(Dropout):  # Hereda de Dropout
    """
    Aplica DropConnect a la capa anterior.
    """

    def __init__(self, rate, **kwargs):
        super(DropConnect, self).__init__(rate, **kwargs)

    def call(self, inputs, training=None):
        if training:
            keep_prob = 1 - self.rate
            shape = (inputs.shape[0],) + (1,) * (len(inputs.shape) - 1)  # (batch_size, 1, 1, ...)
            random_tensor = keep_prob + tf.random.uniform(shape, dtype=inputs.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = (inputs / keep_prob) * binary_tensor
            return output
        else:
            return inputs


# --- Scheduled Drop Path ---
class ScheduledDropPath(Layer):
    def __init__(self, drop_prob: float = 0.0, **kwargs):
        super(ScheduledDropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if training and self.drop_prob > 0.0:
            shape = tf.shape(x)  # Misma forma que la entrada
            keep_prob = 1 - self.drop_prob
            random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
            binary_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * binary_tensor
        else:
            return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({'drop_prob': self.drop_prob})
        return config


# --- Multi-Query Attention (Como alternativa a Multi-Head Attention) ---
class MultiQueryAttention(Layer):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super(MultiQueryAttention, self).__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.depth = d_model // num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)
        self.dropout = Dropout(dropout_rate)

        # Se usa MultiHeadAttention de Keras
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=self.depth)

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor,
             mask: tf.Tensor = None, training: bool = None) -> tf.Tensor:

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)

        # Aplicar atención multi-cabeza con máscara
        attention_output, attention_weights = self.attention(
            query=q, key=k, value=v, attention_mask=mask, return_attention_scores=True
        )

        output = self.dense(attention_output)  # (batch_size, seq_len_q, d_model)
        output = self.dropout(output, training=training)

        return output, attention_weights

    def get_config(self):
        config = super(MultiQueryAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config