# tests/models/cnn_lstm/test_model.py
import pytest
import tensorflow as tf
from models.cnn_lstm.model import CNN_LSTM
from models.cnn_lstm.config import CNNLSTMConfig

# Fixture para una configuración de prueba
@pytest.fixture
def cnn_lstm_config():
    return CNNLSTMConfig(
        cnn_filters=[32, 64],
        cnn_kernel_sizes=[3, 3],
        cnn_pool_sizes=[2, 2],
        cnn_dilation_rates = [1, 2],
        lstm_units=128,
        dropout_rate=0.1,
        dense_units=[64],
        activation='relu',
        l1_reg=0.0,
        l2_reg=0.0,
        use_batchnorm=False,
        use_attention = False #Por defecto
    )

# Prueba de creación del modelo
def test_cnn_lstm_creation(cnn_lstm_config):
    model = CNN_LSTM(config=cnn_lstm_config)
    assert isinstance(model, CNN_LSTM)

# Prueba de la forma de la salida
def test_cnn_lstm_output_shape(cnn_lstm_config):
    model = CNN_LSTM(config=cnn_lstm_config)
    batch_size = 8
    seq_len = 20
    num_features = 10
    inputs = tf.random.normal((batch_size, seq_len, num_features))
    output = model(inputs)
    assert output.shape == (batch_size, 1)  # Expected output shape


# Prueba con entrenamiento activado
def test_cnn_lstm_forward_pass_training(cnn_lstm_config):
    model = CNN_LSTM(config=cnn_lstm_config)
    batch_size = 8
    seq_len = 20
    num_features = 10
    inputs = tf.random.normal((batch_size, seq_len, num_features))
    output = model(inputs, training=True)  # Pass training=True
    assert output.shape == (batch_size, 1)

# Prueba con diferentes configuraciones (parametrización)
@pytest.mark.parametrize("use_batchnorm, dropout_rate", [(True, 0.2), (False, 0.0)])
def test_cnn_lstm_with_config(cnn_lstm_config, use_batchnorm, dropout_rate):
    cnn_lstm_config.use_batchnorm = use_batchnorm
    cnn_lstm_config.dropout_rate = dropout_rate
    model = CNN_LSTM(config=cnn_lstm_config)
    batch_size = 4
    seq_len = 15
    num_features = 8
    inputs = tf.random.normal((batch_size, seq_len, num_features))
    output = model(inputs)
    assert output.shape == (batch_size, 1)

# Prueba de get_config
def test_get_config(cnn_lstm_config):
    model = CNN_LSTM(config=cnn_lstm_config)
    config = model.get_config()
    assert isinstance(config, dict)
    # Add checks to verify that the config contains the expected keys and values.
    assert config['dropout_rate'] == cnn_lstm_config.dropout_rate
    assert config['lstm_units'] == cnn_lstm_config.lstm_units
    assert config['cnn_filters'] == cnn_lstm_config.cnn_filters

#Test con y sin atención
@pytest.mark.parametrize("use_attention", [True, False])
def test_cnn_lstm_attention(cnn_lstm_config, use_attention):
    cnn_lstm_config.use_attention = use_attention #Modificar la configuración existente
    if use_attention: #Si usamos atención
        cnn_lstm_config.attention_heads = 2 #Debemos configurar un numero de cabezas
    model = CNN_LSTM(config=cnn_lstm_config) #Pasar la configuración modificada
    batch_size = 8
    seq_len = 20
    num_features = 10
    inputs = tf.random.normal((batch_size, seq_len, num_features))
    output = model(inputs)
    assert output.shape == (batch_size, 1)  # Expected output shape