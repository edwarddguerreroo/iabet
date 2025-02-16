# tests/models/tft/test_training.py
import pytest
import tensorflow as tf
import numpy as np
# from models.tft.base.model import TFT  # No necesario si ya tienes la factoría
from models.tft.base.training import train_one_epoch, evaluate_epoch, quantile_loss, evidential_loss, mdn_loss
from models.tft.base.config import TFTConfig
from pipelines.preprocessing.preprocess_data import load_and_preprocess_data
# from core.utils.config import load_config, parse_args #Si se usa

# Fixture para una configuración de prueba
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
    #Resto de parámetros
    'use_positional_encoding': False,
    'use_dropconnect': False,
    'use_scheduled_drop_path': False,
    'drop_path_rate': 0.1,
    'kernel_initializer': "glorot_uniform",
    'use_glu_in_grn': True,
    'use_layer_norm_in_grn': True,
    'use_multi_query_attention': False,
    'use_indrnn': False,
    'use_logsparse_attention': False,
    'sparsity_factor': 4,
    'use_evidential_regression': False,
    'use_mdn': False,
    'num_mixtures': 5,
    'use_time2vec': False,
    'time2vec_dim': 32,
    'time2vec_activation': "sin",
    'use_fourier_features': False,
    'num_fourier_features': 10,
    'use_reformer_attention': False,
    'num_buckets': 8,
    'use_sparsemax': False,
    'l1_reg': 0.0,
    'l2_reg': 0.0,
    'use_gnn': False,
    'gnn_embedding_dim': None,
    'use_transformer': False,
    'transformer_embedding_dim': None,
    'loss': "quantile_loss",
    'seq_len': 4 #Modificar
    }
    return TFTConfig(**config_dict)


# Fixture para datos de entrenamiento de prueba
@pytest.fixture
def train_data_fixture(test_config):
    batch_size = 8
    seq_len = test_config.seq_len #Usar seq_len
    num_features = test_config.raw_time_features_dim
    num_static_features = test_config.raw_static_features_dim
    num_cat_time_features = len(test_config.time_varying_categorical_features_cardinalities)
    num_cat_static_features = len(test_config.static_categorical_features_cardinalities)

    time_varying_numeric_inputs = tf.random.normal((batch_size, seq_len, num_features))
    time_varying_categorical_inputs = tf.random.uniform(
        (batch_size, seq_len, num_cat_time_features), minval=0, maxval=10, dtype=tf.int32
    )
    static_numeric_inputs = tf.random.normal((batch_size, num_static_features))
    static_categorical_inputs = tf.random.uniform((batch_size, num_cat_static_features), minval=0, maxval=3, dtype=tf.int32)
    time_inputs = tf.random.uniform((batch_size, seq_len, 1), minval=0, maxval=24, dtype=tf.float32)
    targets = tf.random.normal((batch_size, seq_len, 1))  # Ejemplo para regresión

    return tf.data.Dataset.from_tensor_slices((
        (time_varying_numeric_inputs, time_varying_categorical_inputs, static_numeric_inputs, static_categorical_inputs, time_inputs),
        targets
    )).batch(batch_size) #Se agrega el batch


# Prueba para train_one_epoch
def test_train_one_epoch(test_config, train_data_fixture):
    from models.tft.base.model import TFT
    #Crear una instancia del modelo
    model = TFT(config=test_config)

    #Crear un optimizador
    optimizer = tf.keras.optimizers.Adam()

    #Definir una función de pérdida
    criterion = tf.keras.losses.MeanSquaredError()
    batch_size = test_config.batch_size if hasattr(test_config, 'batch_size') else 4 #Si no se define, usar 4

    #Ejecutar train_one_epoch
    initial_loss = evaluate_epoch(model, train_data_fixture, criterion, batch_size) #Evaluar antes de entrenar
    train_loss = train_one_epoch(model, train_data_fixture, optimizer, criterion, batch_size) #Entrenar por una época
    assert isinstance(train_loss, float)  # Verificar que la pérdida es un flotante
    #Verificar que la perdida disminuye
    final_loss = evaluate_epoch(model, train_data_fixture, criterion, batch_size)
    assert final_loss < initial_loss, "Loss did not decrease after one epoch of training"

# Prueba para evaluate_epoch (similar a train_one_epoch, pero sin gradientes)
def test_evaluate_epoch(test_config, train_data_fixture):
    from models.tft.base.model import TFT
    #Usamos el mismo dataset que para train_one_epoch
    model = TFT(config=test_config)
    criterion = tf.keras.losses.MeanSquaredError()
    batch_size = test_config.batch_size if hasattr(test_config, 'batch_size') else 4  # Si no se define, usar 4

    val_loss = evaluate_epoch(model, train_data_fixture, criterion, batch_size)
    assert isinstance(val_loss, float)

#Pruebas para las funciones de pérdida
def test_quantile_loss():
    y_true = tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)  # (1, 3, 1)
    y_pred = tf.constant([[[0.9, 2.0, 3.1], [1.8, 2.2, 2.5], [2.8, 3.0, 3.3]]], dtype=tf.float32)  # (1, 3, 3)
    quantiles = [0.1, 0.5, 0.9]
    loss = quantile_loss(y_true, y_pred, quantiles)
    assert isinstance(loss.numpy(), float)

# (Opcional) Prueba para evidential_loss y mdn_loss (si los usas)
def test_evidential_loss():
    y_true = tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)
    #Parametros
    gamma = tf.constant([[[1.0], [2.0], [3.0]]], dtype=tf.float32)
    v = tf.constant([[[1.0], [1.0], [1.0]]], dtype=tf.float32)
    alpha = tf.constant([[[2.0], [2.0], [2.0]]], dtype=tf.float32)
    beta = tf.constant([[[1.0], [1.0], [1.0]]], dtype=tf.float32)
    evidential_params = tf.concat([gamma, v, alpha, beta], axis=-1)
    loss = evidential_loss(y_true, evidential_params)
    assert isinstance(loss.numpy(), float)

def test_mdn_loss():
    y_true = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)  # (3, 1)
    #Supongamos 2 mixturas
    pis = tf.constant([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]], dtype=tf.float32)  # (3, 2)
    mus = tf.constant([[0.9, 2.1], [1.8, 2.5], [2.8, 3.2]], dtype=tf.float32)  # (3, 2)
    sigmas = tf.constant([[0.1, 0.2], [0.3, 0.1], [0.2, 0.3]], dtype=tf.float32)  # (3, 2)
    loss = mdn_loss(y_true, pis, mus, sigmas)
    assert isinstance(loss.numpy(), float)