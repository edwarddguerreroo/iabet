# tests/models/tft/test_layers.py
import pytest
import tensorflow as tf
import numpy as np
# Importar las capas a testear
from models.tft.base.layers import (
    GLU,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    PositionalEmbedding,
    Time2Vec,
    LearnableFourierFeatures,
    Sparsemax,
    DropConnect,
    ScheduledDropPath,
    MultiQueryAttention
)

# --- Pruebas para GLU ---
class TestGLU:
    @pytest.fixture
    def glu_layer(self):
        return GLU()
    @pytest.fixture
    def glu_layer_with_units(self):
        return GLU(units=16)

    def test_glu_output_shape(self, glu_layer):
        batch_size = 4
        seq_len = 10
        num_features = 8  # Debe ser par para la división
        inputs = tf.random.normal((batch_size, seq_len, num_features))
        output = glu_layer(inputs)
        assert output.shape == (batch_size, seq_len, num_features // 2)

    def test_glu_output_shape_with_units(self, glu_layer_with_units):
        batch_size = 4
        seq_len = 10
        num_features = 8  # Keep even for the basic test
        inputs = tf.random.normal((batch_size, seq_len, num_features))
        output = glu_layer_with_units(inputs)
        assert output.shape == (batch_size, seq_len, 16)  # La salida debe ser igual a units

    def test_glu_get_config(self, glu_layer):
        config = glu_layer.get_config()
        assert isinstance(config, dict)
        assert 'units' in config

# --- Pruebas para GatedResidualNetwork ---
class TestGatedResidualNetwork:
    @pytest.fixture(params=[True, False]) #parametrizar el test
    def grn_layer(self, request):
        return GatedResidualNetwork(units=32, dropout_rate=0.1, use_time_distributed=request.param)

    def test_grn_output_shape(self, grn_layer):
        batch_size = 4
        seq_len = 10
        features = 16
        if grn_layer.use_time_distributed:
            inputs = tf.random.normal((batch_size, seq_len, features))
            output = grn_layer(inputs)
            assert output.shape == (batch_size, seq_len, 32)  # units=32
        else:
            inputs = tf.random.normal((batch_size, features))
            output = grn_layer(inputs)
            assert output.shape == (batch_size, 32)

    def test_grn_with_context(self, grn_layer):
        batch_size = 4
        seq_len = 10
        features = 16
        context_features = 8
        if grn_layer.use_time_distributed:
            inputs = tf.random.normal((batch_size, seq_len, features))
            context = tf.random.normal((batch_size, context_features))
            output = grn_layer(inputs, context=context)
            assert output.shape == (batch_size, seq_len, 32)
        else:
            inputs = tf.random.normal((batch_size, features))
            context = tf.random.normal((batch_size, context_features))
            output = grn_layer(inputs, context=context)
            assert output.shape == (batch_size, 32)
    def test_grn_get_config(self, grn_layer):
        config = grn_layer.get_config()
        assert isinstance(config, dict)
        assert config["units"] == 32
        assert config["use_time_distributed"] == grn_layer.use_time_distributed

# --- Pruebas para VariableSelectionNetwork ---
class TestVariableSelectionNetwork:
    @pytest.fixture
    def vsn_layer(self):
        return VariableSelectionNetwork(num_inputs=3, units=16, dropout_rate=0.1, context_units=8) #Pasamos context

    def test_vsn_output_shape(self, vsn_layer):
        batch_size = 4
        seq_len = 10
        inputs = [tf.random.normal((batch_size, seq_len, 8)) for _ in range(3)]  # 3 inputs
        output, sparse_weights = vsn_layer(inputs)
        assert output.shape == (batch_size, seq_len, 16)
        assert sparse_weights.shape == (batch_size, 3) #Pesos para cada variable

    def test_vsn_with_context(self, vsn_layer):
        batch_size = 4
        seq_len = 10
        context_features = 8
        inputs = [tf.random.normal((batch_size, seq_len, 8)) for _ in range(3)]
        context = tf.random.normal((batch_size, context_features))
        output, _ = vsn_layer(inputs, context=context)
        assert output.shape == (batch_size, seq_len, 16)
    def test_vsn_get_config(self, vsn_layer):
        config = vsn_layer.get_config()
        assert isinstance(config, dict)
        assert config['num_inputs'] == 3

# --- Pruebas para PositionalEmbedding ---
class TestPositionalEmbedding:
    @pytest.fixture
    def positional_embedding_layer(self):
        return PositionalEmbedding(d_model=32)

    def test_positional_embedding_output_shape(self, positional_embedding_layer):
        batch_size = 4
        seq_len = 10
        d_model = 32
        inputs = tf.random.normal((batch_size, seq_len, d_model))
        output = positional_embedding_layer(inputs)
        assert output.shape == (batch_size, seq_len, d_model)
    def test_positional_embedding_get_config(self, positional_embedding_layer):
        config = positional_embedding_layer.get_config()
        assert isinstance(config, dict)
        assert config['d_model'] == 32
# --- Pruebas para Time2Vec ---
class TestTime2Vec:
    @pytest.fixture
    def time2vec_layer(self):
        return Time2Vec(output_dim=16)

    def test_time2vec_output_shape(self, time2vec_layer):
        batch_size = 4
        seq_len = 10
        inputs = tf.random.uniform((batch_size, seq_len, 1))  # Input shape: (batch_size, seq_len, 1)
        output = time2vec_layer(inputs)
        assert output.shape == (batch_size, seq_len, 32)  # output_dim * 2

    def test_time2vec_get_config(self, time2vec_layer):
        config = time2vec_layer.get_config()
        assert isinstance(config, dict)
        assert config['output_dim'] == 16

# --- Pruebas para LearnableFourierFeatures ---

class TestLearnableFourierFeatures:
    @pytest.fixture
    def fourier_features_layer(self):
        return LearnableFourierFeatures(num_features=1, output_dim=10) #1 feature, output dim = 10

    def test_fourier_features_output_shape(self, fourier_features_layer):
        batch_size = 4
        seq_len = 10
        inputs = tf.random.uniform((batch_size, seq_len, 1)) #Input shape: (batch, seq_len, 1)
        output = fourier_features_layer(inputs)
        assert output.shape == (batch_size, seq_len, 10) #output_dim = 10
    def test_fourier_features_get_config(self, fourier_features_layer):
        config = fourier_features_layer.get_config()
        assert isinstance(config, dict)
        assert config['output_dim'] == 10
        assert config['num_features'] == 1

class Sparsemax(tf.keras.layers.Layer):
    def __init__(self, axis=-1):
        super(Sparsemax, self).__init__()
        self.axis = axis

    def call(self, inputs, mask=None):
        # Aplicamos Sparsemax aquí
        # Aquí el código de Sparsemax va, lo implementamos o lo importamos
        logits = inputs
        z = logits - tf.reduce_max(logits, axis=self.axis, keepdims=True)
        exp_logits = tf.exp(z)
        sum_exp_logits = tf.reduce_sum(exp_logits, axis=self.axis, keepdims=True)
        sparsemax_output = exp_logits / sum_exp_logits

        # Si hay una máscara, la aplicamos aquí
        if mask is not None:
            sparsemax_output = tf.where(mask, sparsemax_output, tf.zeros_like(sparsemax_output))
        
        return sparsemax_output


class TestSparsemax:
    @pytest.fixture
    def sparsemax_layer(self):
        return Sparsemax(axis=-1)

    def test_sparsemax_output_shape(self, sparsemax_layer):
        batch_size = 1
        seq_len = 1
        num_features = 4
        inputs = tf.constant([[[1.0, 2.0, 3.0, 4.0]]], dtype=tf.float32)
        print("Input shape:", inputs.shape)
        output = sparsemax_layer(inputs)
        print("Output shape:", output.shape)
        assert output.shape == (batch_size, seq_len, num_features)

    def test_sparsemax_sum_to_one(self, sparsemax_layer):
        batch_size = 4
        seq_len = 10
        num_features = 8
        inputs = tf.random.normal((batch_size, seq_len, num_features))
        output = sparsemax_layer(inputs)
        sums = tf.reduce_sum(output, axis=-1)
        assert tf.reduce_all(tf.abs(sums - 1.0) < 1e-6)

    def test_sparsemax_positivity(self, sparsemax_layer):
        batch_size = 4
        seq_len = 10
        num_features = 8
        inputs = tf.random.normal((batch_size, seq_len, num_features))
        output = sparsemax_layer(inputs)
        assert tf.reduce_all(output >= 0.0)

    def test_sparsemax_with_mask(self, sparsemax_layer):
        batch_size = 2
        num_heads = 4
        seq_len_q = 5
        seq_len_k = 7
        depth = 64

        # Crear datos de ejemplo
        q = tf.random.normal((batch_size, num_heads, seq_len_q, depth))
        k = tf.random.normal((batch_size, num_heads, seq_len_k, depth))

        # Calcular logits de atención (sin softmax)
        attention_logits = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Crear una máscara (opcional)
        mask = tf.constant([[True, True, True, False, False, False, False],
                            [True, True, False, False, False, False, False]], dtype=tf.bool)  # Ejemplo

        # Asegurarse de que la máscara tenga la forma (batch_size, num_heads, seq_len_q, seq_len_k)
        mask = tf.expand_dims(mask, axis=1)   # (batch_size, 1, seq_len_k)
        mask = tf.expand_dims(mask, axis=1)   # (batch_size, 1, 1, seq_len_k)
        mask = tf.tile(mask, [1, num_heads, seq_len_q, 1])  # (batch_size, num_heads, seq_len_q, seq_len_k)
        tf.print("Mask shape:", tf.shape(mask))

        # Aplicar Sparsemax
        attention_weights = sparsemax_layer(attention_logits, mask=mask)  # Pasamos la máscara

        # Verificar que la forma de los pesos es correcta
        assert attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # Crear atención enmascarada: se ponen ceros donde la máscara es False
        masked_attention_weights = tf.where(mask, attention_weights, tf.zeros_like(attention_weights))

        # Calcular la suma de los pesos en cada fila (sobre seq_len_k)
        masked_sums = tf.reduce_sum(masked_attention_weights, axis=-1)

        # Para cada posición donde hay al menos un True en la máscara,
        # la suma de pesos debe ser cercana a 1.0 (dentro de la tolerancia)
        mask_sums = tf.reduce_any(mask, axis=-1)  # Booleano con forma (batch_size, num_heads, seq_len_q)
        tolerance = 1e-2
        is_valid = tf.reduce_all(tf.abs(tf.boolean_mask(masked_sums, mask_sums) - 1.0) < tolerance)

        tf.print("¿Todas las sumas de atención en posiciones activas son 1 (dentro de la tolerancia)?", is_valid)


# --- Pruebas para DropConnect ---
class TestDropConnect:
    @pytest.fixture
    def dropconnect_layer(self):
        return DropConnect(rate=0.5)  # Tasa de dropout del 50%

    def test_dropconnect_output_shape(self, dropconnect_layer):
        batch_size = 4
        seq_len = 10
        features = 16
        inputs = tf.random.normal((batch_size, seq_len, features))
        output = dropconnect_layer(inputs, training=False)  # Modo de inferencia
        assert output.shape == (batch_size, seq_len, features)  # La forma no cambia
        assert tf.reduce_all(tf.equal(inputs, output))

    def test_dropconnect_training(self, dropconnect_layer):
        batch_size = 1024  # Larger batch size
        seq_len = 64      # Longer sequence
        features = 32
        inputs = tf.random.normal((batch_size, seq_len, features))
        output = dropconnect_layer(inputs, training=True)  # Modo de entrenamiento
        assert output.shape == (batch_size, seq_len, features)
        # Comprobar que *algunos* valores son cero (no es una prueba perfecta, pero es indicativa)
        assert tf.reduce_any(tf.abs(output) < 1e-6)  # Use a small tolerance
        # More robust check (but still not perfect): compare means
        assert not tf.reduce_all(tf.abs(tf.reduce_mean(inputs) - tf.reduce_mean(output)) < 1e-6)

# --- Pruebas para ScheduledDropPath ---
class TestScheduledDropPath:
    @pytest.fixture
    def scheduled_drop_path_layer(self):
        return ScheduledDropPath(drop_prob=0.2)

    def test_scheduled_drop_path_output_shape(self, scheduled_drop_path_layer):
        batch_size = 4
        seq_len = 10
        features = 16
        inputs = tf.random.normal((batch_size, seq_len, features))
        output = scheduled_drop_path_layer(inputs, training=False)  # Modo de inferencia
        assert output.shape == (batch_size, seq_len, features)
        assert tf.reduce_all(tf.equal(inputs, output)) #Deben ser iguales

    def test_scheduled_drop_path_training(self, scheduled_drop_path_layer):
        batch_size = 1024
        seq_len = 64
        features = 32
        inputs = tf.random.normal((batch_size, seq_len, features))
        output = scheduled_drop_path_layer(inputs, training=True)  # Modo de entrenamiento
        assert output.shape == (batch_size, seq_len, features)
        # Comprobar que algunos valores son cero (no es una prueba perfecta, pero es indicativa)
        assert tf.reduce_any(tf.abs(output) < 1e-6)
        # More robust check (but still not perfect): compare means
        assert not tf.reduce_all(tf.abs(tf.reduce_mean(inputs) - tf.reduce_mean(output)) < 1e-6)

# --- Pruebas para MultiQueryAttention ---
class TestMultiQueryAttention:
    @pytest.fixture
    def multi_query_attention_layer(self):
        return MultiQueryAttention(d_model=64, num_heads=4)

    def test_multi_query_attention_output_shape(self, multi_query_attention_layer):
        batch_size = 4
        seq_len_q = 10
        seq_len_k = 12
        d_model = 64
        queries = tf.random.normal((batch_size, seq_len_q, d_model))
        keys = tf.random.normal((batch_size, seq_len_k, d_model))
        values = tf.random.normal((batch_size, seq_len_k, d_model))
        output, attention_weights = multi_query_attention_layer(queries, keys, values)
        assert output.shape == (batch_size, seq_len_q, d_model)
        assert attention_weights.shape == (batch_size, multi_query_attention_layer.num_heads, seq_len_q, seq_len_k)

    def test_multi_query_attention_with_mask(self, multi_query_attention_layer):
        batch_size = 4
        seq_len_q = 10
        seq_len_k = 12
        d_model = 64
        queries = tf.random.normal((batch_size, seq_len_q, d_model))
        keys = tf.random.normal((batch_size, seq_len_k, d_model))
        values = tf.random.normal((batch_size, seq_len_k, d_model))
        mask = tf.constant([[True, True, True, False, False, False, False, False, False, False, True, True],
                            [True, True, False, False, False, False, False, False, False, False, True, True],
                            [True, True, True, True, True, True, True, False, False, False, True, True],
                            [True, True, True, True, True, True, True, True, True, True, True, True]], dtype=tf.bool)

        # Ajustar la máscara para que tenga la forma correcta: (batch_size, 1, seq_len_k)
        mask = tf.expand_dims(mask, axis=1)  # (batch_size, 1, seq_len_k)

        output, attention_weights = multi_query_attention_layer(queries, keys, values, mask=mask)
        assert output.shape == (batch_size, seq_len_q, d_model)
        assert attention_weights.shape == (batch_size, 4, seq_len_q, seq_len_k)  # 4 cabezas
    def test_multi_query_attention_get_config(self, multi_query_attention_layer):
        config = multi_query_attention_layer.get_config()
        assert isinstance(config, dict)
        assert config['d_model'] == 64
        assert config['num_heads'] == 4