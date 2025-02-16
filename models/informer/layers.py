import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, MaxPool1D, Dropout, Dense, LayerNormalization
import numpy as np
from typing import Optional, Tuple, List, Union, Callable
#Importar sparsemax desde tft
from models.tft.base.layers import Sparsemax

class ProbSparseAttention(Layer):
    def __init__(self, factor: int = 5, scale: Optional[float] = None, attention_dropout: float = 0.1,
                 output_attention: bool = False, use_sparsemax: bool = False, **kwargs):
        super(ProbSparseAttention, self).__init__(**kwargs)
        self.factor = factor  # Factor de muestreo (C en el paper)
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = Dropout(attention_dropout)
        self.use_sparsemax = use_sparsemax  # Usar Sparsemax
        if use_sparsemax:
            self.sparsemax = Sparsemax(axis=-1)

    def build(self, input_shape):
        super(ProbSparseAttention, self).build(input_shape)

    def _prob_QK(self, Q: tf.Tensor, K: tf.Tensor, sample_k: int, top_k_indices: tf.Tensor) -> Tuple[
        tf.Tensor, tf.Tensor]:
        # Q: [B, H, L, D]
        # K: [B, H, S, D]   (S: longitud de la secuencia de K)

        # Muestreamos K
        B, H, L, D = Q.shape
        _, _, S, _ = K.shape  # Obtenemos la longitud de secuencia de K
        K_sample = tf.gather(K, top_k_indices, axis=2, batch_dims=2)  # Muestreamos K usando los indices
        Q_K = tf.einsum("bhld,bhsd->bhls", Q, K_sample)  # Solo con los K seleccionados
        return Q_K, K_sample  # Devolvemos muestra de K

    def _get_initial_context(self, V: tf.Tensor, L_Q: int) -> tf.Tensor:
        # V: [B, H, S, D]
        # L_Q: longitud de secuencia de Q

        B, H, S, D = V.shape
        if S > L_Q:  # Si S > L_Q, necesitamos hacer un average pooling
            # Necesitamos un stride y un kernel size para el average pooling
            kernel_size = int(np.ceil(S / L_Q))  # Redondeamos hacia arriba
            stride = kernel_size  # Sin overlapping
            context = tf.nn.avg_pool(V, ksize=[1, 1, kernel_size, 1], strides=[1, 1, stride, 1], padding='VALID',
                                     data_format='NHWC')
            # context: (B, H, L', D), donde L' <= L_Q
            # Si L' < L_Q, replicar
            if context.shape[2] < L_Q:
                padding = tf.ones([B, H, L_Q - context.shape[2], D])
                context = tf.concat([context, padding], axis=2)
        else:  # Si no
            context = V  # Se usa V tal cual

        return context  # (B, H, L_Q, D)

    def _update_context(self, context_in: tf.Tensor,
                        values: tf.Tensor,
                        attn_mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        # context_in: [B, H, L_Q, D]
        # values: [B, H, L_Q, D]  <- NOTA: Este 'values' ya NO son los valores originales (V),
        #                             sino el resultado de Q * K_sample.T
        if self.use_sparsemax:
            attn = self.sparsemax(values, mask=attn_mask)  # Usar Sparsemax
        else:
            attn = tf.nn.softmax(values, axis=-1)  # Softmax
        context_in = tf.einsum("bhlk,bhkd->bhld", attn, values)
        return context_in, attn

    def call(self, inputs: List[tf.Tensor],
             attn_mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:

        queries, keys, values = inputs
        B, L, H, E = queries.shape  # Batch, Length_Q, Heads, Embedding_dim
        _, S, _, _ = keys.shape  # _ , Length_K, _ , _

        queries = tf.reshape(queries, (B, H, L, E))
        keys = tf.reshape(keys, (B, H, S, E))
        values = tf.reshape(values, (B, H, S, E))

        # --- Calcular la Medida de Dispersión (Sparsity Measure) ---
        U = self.factor * int(np.ceil(np.log(L)))  # L_K (num keys a muestrear)
        u = self.factor * int(np.ceil(np.log(S)))  # Num de queries

        # Muestreo aleatorio de las queries
        scores_top, top_k_indices = tf.math.top_k(tf.random.uniform((B, H, L)),
                                                   k=min(u, L))  # Indices aleatorios
        Q_sample = tf.gather(queries, top_k_indices, axis=2, batch_dims=2)  # (B, H, u, E) Muestreamos

        # --- Calcular QK (solo con la muestra) ---
        Q_K, K_sample = self._prob_QK(Q_sample, keys, sample_k=U, top_k_indices=top_k_indices)
        # Q_K: (B, H, L, u) -  Atención entre cada query y las *u* keys muestreadas

        # --- Calcular la Medida M(q, K) ---
        M = tf.reduce_max(Q_K, axis=-1) - tf.reduce_mean(Q_K, axis=-1)  # (B, H, L)
        # M: (B, H, L) -  Medida de "dominancia" de cada query

        # --- Seleccionar las Top-U Queries ---
        _, top_queries = tf.math.top_k(M, k=U, sorted=True)  # (B, H, U) - Índices de las top U queries

        # --- Para cada head y batch, expandir los top_queries para obtener los valores de Q, K, V ---
        Q = tf.gather(queries, top_queries, axis=2, batch_dims=2)  # (B, H, U, E)

        # --- Calcular la Atención Final (ProbSparse) ---
        context = self._get_initial_context(values, L)  # (B, H, L, D)
        attn_out, attn = self._update_context(context, tf.matmul(Q, tf.transpose(keys, perm=[0, 1, 3, 2])), attn_mask)
        # attn_out: (B, H, L, D)
        attn_out = tf.reshape(attn_out, (B, L, -1))  # Volver a la forma original

        if self.output_attention:
            return attn_out, attn
        return attn_out, None
    def get_config(self):
        config = super(ProbSparseAttention, self).get_config()
        config.update({
            "factor": self.factor,
            "scale": self.scale,
            "attention_dropout": self.dropout.rate,
            "output_attention": self.output_attention,
            "use_sparsemax": self.use_sparsemax
        })
        return config

class Distilling(Layer):
    def __init__(self, conv_kernel_size: int = 25, out_channels: int = 512, **kwargs):
        super(Distilling, self).__init__(**kwargs)
        # Convolucion
        self.conv = Conv1D(filters=out_channels, kernel_size=conv_kernel_size, padding='same',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))  # Misma longitud
        # Max Pooling
        self.max_pool = MaxPool1D(pool_size=3, strides=2, padding='same')

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # inputs: (batch_size, seq_len, features)
        x = self.conv(inputs)  # (batch_size, seq_len, out_channels)
        x = tf.nn.elu(x)  # ELU
        x = self.max_pool(x)  # (batch_size, seq_len // 2, out_channels)  - Reducir la longitud
        return x
    def get_config(self):
        config = super(Distilling, self).get_config()
        config.update({
            "conv_kernel_size": self.conv.kernel_size[0], #Guardar
            "out_channels": self.conv.filters
        })
        return config

# --- Encoder Layer ---
class EncoderLayer(Layer):
    def __init__(self, attention: Layer, d_model: int, d_ff: Optional[int] = None, dropout_rate: float = 0.1,
                 activation: Union[str, Callable] = "relu", l1_reg: float = 0.0, l2_reg: float = 0.0, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.attention = attention
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.get(activation)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.conv1 = Conv1D(filters=self.d_ff, kernel_size=1,
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))  # Regularizacion
        self.conv2 = Conv1D(filters=self.d_model, kernel_size=1,
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)

    def call(self, x: tf.Tensor, attn_mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        # x: (batch_size, seq_len, d_model)
        attn_output, attn = self.attention([x, x, x], attn_mask=attn_mask, training=training)  # Self-attention
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(x + attn_output)  # Add & Norm
        ffn_output = self.conv2(self.activation(self.conv1(out1)))
        ffn_output = self.dropout(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)  # Add & Norm
        return out2, attn
    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        #Guardar configuración de la atención
        config['attention'] = tf.keras.utils.serialize_keras_object(self.attention)
        return config

# --- Decoder Layer ---
class DecoderLayer(Layer):
    def __init__(self, self_attention: Layer, cross_attention: Layer, d_model: int, d_ff: Optional[int] = None,
                 dropout_rate: float = 0.1, activation: Union[str, Callable] = "relu",
                 l1_reg: float = 0.0, l2_reg: float = 0.0,
                 **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        self.dropout_rate = dropout_rate
        self.activation = tf.keras.activations.get(activation)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        self.conv1 = Conv1D(filters=self.d_ff, kernel_size=1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg)) #Regularizacion
        self.conv2 = Conv1D(filters=self.d_model, kernel_size=1, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=l2_reg))
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.norm3 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)


    def call(self, x: tf.Tensor, enc_out: tf.Tensor, attn_mask: Optional[tf.Tensor] = None,
             cross_attn_mask: Optional[tf.Tensor] = None, training: Optional[bool] = None) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        # x: (batch_size, target_seq_len, d_model)
        # enc_out: (batch_size, input_seq_len, d_model)

        # --- Self-Attention ---
        self_attn_output, self_attn = self.self_attention([x, x, x], attn_mask=attn_mask, training=training)
        self_attn_output = self.dropout(self_attn_output, training=training)
        out1 = self.norm1(x + self_attn_output)

        # --- Cross-Attention (entre decoder y encoder) ---
        cross_attn_output, cross_attn = self.cross_attention([out1, enc_out, enc_out], attn_mask=cross_attn_mask, training=training)
        cross_attn_output = self.dropout(cross_attn_output, training=training)
        out2 = self.norm2(out1 + cross_attn_output)

        # --- Feed Forward ---
        ffn_output = self.conv2(self.activation(self.conv1(out2)))
        ffn_output = self.dropout(ffn_output, training=training)
        out3 = self.norm3(out2 + ffn_output)  # Add & Norm

        return out3, self_attn, cross_attn
    def get_config(self):
        config = super(DecoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg
        })
        #Guardar configuración de las atenciones
        config['self_attention'] = tf.keras.utils.serialize_keras_object(self.self_attention)
        config['cross_attention'] = tf.keras.utils.serialize_keras_object(self.cross_attention)
        return config