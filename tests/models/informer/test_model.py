# tests/models/informer/test_model.py
import pytest
import tensorflow as tf
from models.informer.model import Informer
from models.informer.config import InformerConfig
from core.utils.helpers import load_config  # Assuming you have load_config

# Fixture for a test configuration (you can use a YAML file or a dictionary)
@pytest.fixture
def test_config():
    # Use a dictionary for more control in tests
    config_dict = {
      "enc_in": 7,
      "dec_in": 7,
      "c_out": 7,
      "seq_len": 96,
      "label_len": 48,
      "out_len": 24,
      "factor": 5,
      "d_model": 16, #Reduce for faster testing
      "n_heads": 4,
      "e_layers": 2,
      "d_layers": 1,
      "d_ff": 32,
      "dropout_rate": 0.1,
      "activation": "relu",
      "output_attention": False,
      "distil": True,
      "mix": True,
      "embed_type": 'fixed',
      "freq": 'h',
      "embed_positions": True,
      "conv_kernel_size": 25,
      #Nuevos parámetros
        "use_time2vec": False,
        "time2vec_dim": 16,
        "use_fourier_features": False,
        "num_fourier_features": 10,
        "use_sparsemax": False,
        "use_indrnn": False,
        "use_logsparse_attention": True,
        "l1_reg": 0.0,
        "l2_reg": 0.0,
        "use_scheduled_drop_path": False,
        "drop_path_rate": 0.1,
        "use_gnn": False,
        "gnn_embedding_dim": None,
        "use_transformer": False,
        "transformer_embedding_dim": None,
        "time2vec_activation": "sin"
    }
    return InformerConfig(**config_dict)

# Test model creation
def test_informer_creation(test_config):
    model = Informer(config=test_config)
    assert isinstance(model, Informer)

# Test output shape
def test_informer_output_shape(test_config):
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 4), minval=0, maxval=24, dtype=tf.float32) #4 features
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 4), minval=0, maxval=24, dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    output = model(inputs)
    assert output.shape == (batch_size, test_config.out_len, test_config.c_out)

# Test with training=True
def test_informer_forward_pass_training(test_config):
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 4), minval=0, maxval=24, dtype=tf.float32)
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 4), minval=0, maxval=24, dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    output = model(inputs, training=True)  # Pass training=True
    assert output.shape == (batch_size, test_config.out_len, test_config.c_out)

# Test with output_attention=True
def test_informer_output_attention(test_config):
    test_config.output_attention = True  # Enable attention output
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 4), minval=0, maxval=24, dtype=tf.float32)
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 4), minval=0, maxval=24, dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    output, attns = model(inputs)  # Get both outputs
    assert isinstance(output, tf.Tensor)
    assert output.shape == (batch_size, test_config.out_len, test_config.c_out)
    assert isinstance(attns, list)  # attns should be a list

#Prueba con Time2Vec
def test_informer_time2vec(test_config):
    test_config.use_time2vec = True #Se activa
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 1), minval=0, maxval=24, dtype=tf.float32)
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 1), minval=0, maxval=24, dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, test_config.out_len, test_config.c_out)

#Prueba con Fourier Features
def test_informer_fourier_features(test_config):
    test_config.use_fourier_features = True #Se activa
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 1), minval=0, maxval=24, dtype=tf.float32)
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 1), minval=0, maxval=24, dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, test_config.out_len, test_config.c_out)

#Prueba con Sparsemax
def test_informer_sparsemax(test_config):
    test_config.use_sparsemax = True #Se activa
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 1), minval=0, maxval=24, dtype=tf.float32)
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 1), minval=0, maxval=24, dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    output = model(inputs, training=True)
    assert output.shape == (batch_size, test_config.out_len, test_config.c_out)

#Prueba de guardado y carga
def test_informer_save_load(test_config, tmp_path): #tmp_path es un fixture de pytest
    model = Informer(config=test_config)
    filepath = str(tmp_path / "test_model")  # Convertir a string
    model.save(filepath)
    loaded_model = Informer() #Crear instancia vacia
    loaded_model.load(filepath)
    assert isinstance(loaded_model, Informer)
    #Verificar que los config son iguales
    assert model.config == loaded_model.config
#Prueba de pesos de atencion
def test_informer_get_attention_weights(test_config):
    model = Informer(config=test_config)
    batch_size = 8
    enc_input = tf.random.normal((batch_size, test_config.seq_len, test_config.enc_in))
    dec_input = tf.random.normal((batch_size, test_config.label_len + test_config.out_len, test_config.dec_in))
    enc_time = tf.random.uniform((batch_size, test_config.seq_len, 4), minval=0, maxval=24, dtype=tf.float32)
    dec_time = tf.random.uniform((batch_size, test_config.label_len + test_config.out_len, 4), minval=0, maxval=24,
                                 dtype=tf.float32)
    inputs = (enc_input, dec_input, enc_time, dec_time)
    enc_attns, dec_attns, cross_attns = model.get_attention_weights(inputs)
    assert isinstance(enc_attns, list)
    assert isinstance(dec_attns, list)
    assert isinstance(cross_attns, list)