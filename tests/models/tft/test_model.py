# tests/models/tft/test_model.py
import pytest
import tensorflow as tf
import numpy as np
from models.tft.base.model import TFT
from models.tft.base.config import TFTConfig
from models.tft.base.layers import *  # Import all layers for individual testing


# Fixture for a test configuration
@pytest.fixture
def test_config():
    # Use a dictionary for easier control in tests
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
        "use_positional_encoding": False,
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
        "seq_len": 12
    }
    return TFTConfig(**config_dict)

# Test model creation
def test_tft_creation(test_config):
    model = TFT(config=test_config)
    assert isinstance(model, TFT)

# Test output shape
def test_tft_output_shape(test_config):
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    time_varying_numeric_inputs = tf.random.normal((batch_size, seq_len, test_config.raw_time_features_dim))
    time_varying_categorical_inputs = tf.random.uniform(
        (batch_size, seq_len, len(test_config.time_varying_categorical_features_cardinalities)),
        minval=0, maxval=10, dtype=tf.int32
    )
    static_numeric_inputs = tf.random.normal((batch_size, test_config.raw_static_features_dim))
    static_categorical_inputs = tf.random.uniform((batch_size, len(test_config.static_categorical_features_cardinalities)), minval=0, maxval=3, dtype=tf.int32)
    time_inputs = tf.random.uniform((batch_size, seq_len, 1), minval=0, maxval=24, dtype=tf.float32)

    inputs = (time_varying_numeric_inputs, time_varying_categorical_inputs,
              static_numeric_inputs, static_categorical_inputs, time_inputs)
    output = model(inputs)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

# Test with training=True
def test_tft_forward_pass_training(test_config):
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    time_varying_numeric_inputs = tf.random.normal((batch_size, seq_len, test_config.raw_time_features_dim))
    time_varying_categorical_inputs = tf.random.uniform(
        (batch_size, seq_len, len(test_config.time_varying_categorical_features_cardinalities)),
        minval=0, maxval=10, dtype=tf.int32
    )
    static_numeric_inputs = tf.random.normal((batch_size, test_config.raw_static_features_dim))
    static_categorical_inputs = tf.random.uniform((batch_size, len(test_config.static_categorical_features_cardinalities)), minval=0, maxval=3, dtype=tf.int32)
    time_inputs = tf.random.uniform((batch_size, seq_len, 1), minval=0, maxval=24, dtype=tf.float32)
    inputs = (time_varying_numeric_inputs, time_varying_categorical_inputs,
              static_numeric_inputs, static_categorical_inputs, time_inputs)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)


def test_tft_positional_encoding(test_config):
    test_config.use_positional_encoding = True
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

def test_tft_time2vec(test_config):
    test_config.use_time2vec = True
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

def test_tft_fourier_features(test_config):
    test_config.use_fourier_features = True
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)


def test_tft_indrnn(test_config):
    test_config.use_indrnn = True
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)

def test_tft_logsparse(test_config):
    test_config.use_logsparse_attention = True
    model = TFT(config=test_config)
    batch_size = 8
    seq_len = 12
    inputs = create_dummy_inputs(test_config, batch_size, seq_len)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, seq_len, test_config.num_quantiles)
# Helper function to create dummy inputs based on the config
def create_dummy_inputs(config, batch_size, seq_len):
  time_varying_numeric_inputs = tf.random.normal((batch_size, seq_len, config.raw_time_features_dim))
  time_varying_categorical_inputs = tf.random.uniform(
      (batch_size, seq_len, len(config.time_varying_categorical_features_cardinalities)),
      minval=0, maxval=10, dtype=tf.int32
  )
  static_numeric_inputs = tf.random.normal((batch_size, config.raw_static_features_dim))
  static_categorical_inputs = tf.random.uniform((batch_size, len(config.static_categorical_features_cardinalities)), minval=0, maxval=3, dtype=tf.int32)
  time_inputs = tf.random.uniform((batch_size, seq_len, 1), minval=0, maxval=24, dtype=tf.float32)
  return (time_varying_numeric_inputs, time_varying_categorical_inputs,
            static_numeric_inputs, static_categorical_inputs, time_inputs)

# Prueba de get_config
def test_get_config(test_config):
    model = TFT(config=test_config)
    config = model.get_config()
    assert isinstance(config, dict)
    # Puedes añadir más verificaciones aquí para asegurarte de que la configuración
    # contiene los valores esperados