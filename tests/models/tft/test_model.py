# tests/models/tft/test_model.py
import pytest
import tensorflow as tf
import numpy as np
from models.tft.base.model import TFT
from models.tft.base.config import TFTConfig
#Importar capas
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

# Fixture para una configuración de prueba (puedes usar un archivo YAML o un diccionario)
@pytest.fixture
def test_config():
    # Usar un diccionario para mayor control en las pruebas
    config_dict = {
        "raw_time_features_dim": 5,
        "raw_static_features_dim": 2,
        "time_varying_categorical_features_cardinalities": [10, 5],
        "static_categorical_features_cardinalities": [3],
        "num_quantiles": 3,
        "hidden_size": 16,
        "lstm_layers": 1,
        "attention_heads": 2,
        "dropout_rate": 0.1,
        "use_positional_encoding": False, #Probar con False
        "use_dropconnect": False,
        "use_scheduled_drop_path": False,
        "drop_path_rate": 0.1,
        "kernel_initializer": "glorot_uniform",
        "use_glu_in_grn": True,
        "use_layer_norm_in_grn": True,
        "use_multi_query_attention": False,
        "use_indrnn": False,
        "use_logsparse_attention": False,
        "sparsity_factor": 4,
        "use_evidential_regression": False,
        "use_mdn": False,
        "num_mixtures": 5,
        "use_time2vec": False,
        "time2vec_dim": 32,
        'time2vec_activation': 'sin',
        "use_fourier_features": False,
        "num_fourier_features": 10,
        "use_reformer_attention": False,
        "num_buckets": 8,
        "use_sparsemax": False,
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "use_gnn": False,
        "gnn_embedding_dim": None,
        "use_transformer": False,
        "transformer_embedding_dim": None,
        "loss": "quantile_loss",
        "seq_len": 12 #Agregamos seq_len
    }
    return TFTConfig(**config_dict)

# Prueba de creación del modelo
def test_tft_creation(test_config):
    model = TFT(config=test_config)
    assert isinstance(model, TFT)

# Prueba de la forma de la salida
def test_tft_output_shape(test_config):
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = test_config.seq_len
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    
    print(f"Inputs: {len(inputs)} elementos")
    for i, tensor in enumerate(inputs):
        print(f"Tensor {i}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")

# Prueba con entrenamiento activado
def test_tft_forward_pass_training(test_config):
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = test_config.seq_len
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)

    print(f"🔹 Número de tensores en inputs: {len(inputs)}")
    for i, tensor in enumerate(inputs):
        print(f"🔹 Tensor {i} shape: {tensor.shape}, dtype: {tensor.dtype}")

    #  Detectar errores antes de pasar los inputs al modelo
    #  Agregar un tercer tensor temporalmente para ver si el error cambia
    static_inputs = [inputs[2], inputs[3], inputs[4]]  # Asegura que sean 3 tensores reales
    print(f"🛠️ static_inputs antes de vsn_static: {static_inputs}")
    
    # Intentamos el forward pass
    output = model([inputs[0], inputs[1], *static_inputs], training=True)
    print(f" Output shape: {output.shape}")


#Prueba con positional encoding
def test_tft_positional_encoding(test_config):
    test_config.use_positional_encoding = True #Se activa
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len) #Se crean con la función
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

#Prueba con Time2Vec
def test_tft_time2vec(test_config):
    test_config.use_time2vec = True #Se activa
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len) #Se crean con la función
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

#Prueba con Fourier Features
def test_tft_fourier_features(test_config):
    test_config.use_fourier_features = True #Se activa
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len) #Se crean con la función
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

#Prueba usando IndRNN
def test_tft_indrnn(test_config):
    test_config.use_indrnn = True  # Se activa
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len) #Se crean con la función
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

#Prueba usando LogSparse
def test_tft_logsparse(test_config):
    test_config.use_logsparse_attention = True  # Se activa
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)
# Prueba de get_config
def test_get_config(test_config):
    model = TFT(config=test_config)
    config = model.get_config()
    assert isinstance(config, dict)
    # Puedes añadir más verificaciones aquí para asegurarte de que la configuración
    # contiene los valores esperados

# Helper function to create dummy inputs based on the config
def create_dummy_inputs(config, batch_size, seq_len):
    # 🔹 Crear características de tiempo variantes (numéricas y categóricas)
    time_varying_numeric_inputs = tf.random.normal((batch_size, seq_len, config.raw_time_features_dim))

    #  **Generar categorías dentro del rango correcto**
    time_varying_categorical_inputs = [
        tf.random.uniform((batch_size, seq_len, 1),
                          minval=0,
                          maxval=config.time_varying_categorical_features_cardinalities[i],  # Respetar rango
                          dtype=tf.int32)
        for i in range(len(config.time_varying_categorical_features_cardinalities))
    ]
    time_varying_categorical_inputs = tf.stack(time_varying_categorical_inputs, axis=0)

    # 🔹 Crear características estáticas
    static_numeric_inputs = tf.random.normal((batch_size, config.raw_static_features_dim))
    static_categorical_inputs = tf.random.uniform((batch_size, 1), minval=0, maxval=5, dtype=tf.int32)

    # 🔹 Crear entradas de tiempo
    time_inputs = tf.random.normal((batch_size, seq_len, 1))

    return (
        time_varying_numeric_inputs,
        time_varying_categorical_inputs,
        static_numeric_inputs,
        static_categorical_inputs,
        time_inputs,
    )

# Prueba de cantidad de entradas

def test_input_tensor_count(test_config):
    batch_size = 8
    seq_len = test_config.seq_len
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    assert len(inputs) == 5, f"Se esperaban 5 entradas, pero se obtuvieron {len(inputs)}"