import pytest
import tensorflow as tf
from models.informer.layers import ProbSparseAttention, Distilling, EncoderLayer, DecoderLayer
import numpy as np

# --- Pruebas para ProbSparseAttention ---
class TestProbSparseAttention:
    @pytest.fixture
    def attention_layer(self):
        return ProbSparseAttention(factor=5, attention_dropout=0.1)

    def test_prob_qk(self, attention_layer):
        batch_size = 4
        heads = 8
        L = 20  # Longitud de Q
        S = 30  # Longitud de K (puede ser diferente)
        E = 16  # Dimensión de embedding
        Q = tf.random.normal((batch_size, heads, L, E))
        K = tf.random.normal((batch_size, heads, S, E))
        sample_k = 5 #Tomamos 5 como ejemplo
        top_k_indices = tf.random.uniform((batch_size, heads, L), minval=0, maxval= S, dtype=tf.int32) #Indices aleatorios
        Q_K, K_sample = attention_layer._prob_QK(Q, K, sample_k, top_k_indices)
        assert Q_K.shape == (batch_size, heads, L, sample_k)  # Verificar la forma de Q_K
        assert K_sample.shape == (batch_size, heads, sample_k, E) #Verificar la forma de K

    def test_get_initial_context(self, attention_layer):
        batch_size = 4
        heads = 8
        S = 30
        D = 16
        L_Q = 20
        V = tf.random.normal((batch_size, heads, S, D))
        context = attention_layer._get_initial_context(V, L_Q)
        assert context.shape == (batch_size, heads, L_Q, D)

    #Prueba con S < L_Q
    def test_get_initial_context_short_S(self, attention_layer):
        batch_size = 4
        heads = 8
        S = 10  # S menor que L_Q
        D = 16
        L_Q = 20
        V = tf.random.normal((batch_size, heads, S, D))
        context = attention_layer._get_initial_context(V, L_Q)
        assert context.shape == (batch_size, heads, L_Q, D) #Debe ser L_Q

    def test_update_context(self, attention_layer):
        batch_size = 4
        heads = 8
        L_Q = 20
        D = 16
        context_in = tf.random.normal((batch_size, heads, L_Q, D))
        values = tf.random.normal((batch_size, heads, L_Q, D))
        context_out, attn = attention_layer._update_context(context_in, values)
        assert context_out.shape == (batch_size, heads, L_Q, D)
        assert attn.shape == (batch_size, heads, L_Q, L_Q) #Matriz de atencion

    def test_call(self, attention_layer):
        batch_size = 4
        L_Q = 20
        S = 30
        heads = 8
        D = 16
        queries = tf.random.normal((batch_size, L_Q, heads, D))
        keys = tf.random.normal((batch_size, S, heads, D))
        values = tf.random.normal((batch_size, S, heads, D))
        inputs = [queries, keys, values]
        output, attn = attention_layer(inputs)
        assert output.shape == (batch_size, L_Q, heads * D)
        assert attn is not None #Debe retornar atención
        #Prueba sin retornar atencion
        attention_layer.output_attention = False
        output, attn = attention_layer(inputs)
        assert attn is None
    def test_attention_sparsemax(self):
        # Crear una instancia de la capa ProbSparseAttention con use_sparsemax=True
        attention_layer = ProbSparseAttention(factor=5, use_sparsemax=True)

        # Crear datos de entrada de prueba
        batch_size = 2
        heads = 4
        L_Q = 5
        S = 7
        D = 8
        queries = tf.random.normal((batch_size, L_Q, heads, D))
        keys = tf.random.normal((batch_size, S, heads, D))
        values = tf.random.normal((batch_size, S, heads, D))

        # Crear una máscara de atención (opcional)
        attn_mask = tf.constant([[True, True, True, False, False, False, False],
                                 [True, True, False, False, False, False, False]],
                                dtype=tf.bool)
        attn_mask = tf.expand_dims(attn_mask, axis=1)  # (batch_size, seq_len_k) -> (batch_size, 1, seq_len_k)
        attn_mask = tf.expand_dims(attn_mask, axis=1) #  (batch_size, 1, 1, seq_len_k) -> Para una sola cabeza
        # attn_mask = tf.tile(attn_mask, [1, heads, L_Q, 1]) #Ya no se necesita

        # Llamar al método call de la capa
        output, attn_weights = attention_layer([queries, keys, values], attn_mask=attn_mask)

        # Verificar la forma de la salida
        assert output.shape == (batch_size, L_Q, heads * D)

        # Verificar que los pesos de atención suman 1 (o están enmascarados)
        if attn_weights is not None:
             sums = tf.reduce_sum(attn_weights, axis=-1)  # (batch_size, num_heads, seq_len_q)
             mask_sums = tf.reduce_any(attn_mask, axis=-1)  # (batch_size, num_heads, seq_len_q) -> Algun True por fila
             assert tf.reduce_all(tf.abs(tf.boolean_mask(sums, mask_sums) - 1.0) < 1e-6)

# --- Pruebas para Distilling ---
class TestDistilling:
    @pytest.fixture
    def distilling_layer(self):
        return Distilling(conv_kernel_size=3, out_channels=64)

    def test_call(self, distilling_layer):
        batch_size = 8
        seq_len = 20
        features = 32
        inputs = tf.random.normal((batch_size, seq_len, features))
        output = distilling_layer(inputs)
        # Verificar que la longitud de la secuencia se ha reducido (debido al max pooling)
        assert output.shape[1] == seq_len // 2 + (seq_len % 2)
        assert output.shape[2] == 64  # Verificar el número de canales de salida

# --- Pruebas para EncoderLayer ---
class TestEncoderLayer:
    @pytest.fixture
    def encoder_layer(self):
        attention = ProbSparseAttention(factor=5)  # Usar ProbSparseAttention
        return EncoderLayer(attention, d_model=32, d_ff=64)

    def test_call(self, encoder_layer):
        batch_size = 4
        seq_len = 10
        d_model = 32
        inputs = tf.random.normal((batch_size, seq_len, d_model))
        output, attn = encoder_layer(inputs)
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn is not None

# --- Pruebas para DecoderLayer ---
class TestDecoderLayer:
    @pytest.fixture
    def decoder_layer(self):
        self_attention = ProbSparseAttention(factor=5)
        cross_attention = ProbSparseAttention(factor=5)
        return DecoderLayer(self_attention, cross_attention, d_model=32, d_ff=64)

    def test_call(self, decoder_layer):
        batch_size = 4
        target_seq_len = 8
        input_seq_len = 10
        d_model = 32
        x = tf.random.normal((batch_size, target_seq_len, d_model))
        enc_out = tf.random.normal((batch_size, input_seq_len, d_model))
        output, self_attn, cross_attn = decoder_layer(x, enc_out)
        assert output.shape == (batch_size, target_seq_len, d_model)
        assert self_attn is not None
        assert cross_attn is not None