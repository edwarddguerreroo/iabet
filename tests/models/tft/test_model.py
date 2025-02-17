# tests/models/tft/test_model.py
import pytest
import tensorflow as tf
import numpy as np
from models.tft.base.model import TFT, create_dummy_inputs
from models.tft.base.config import TFTConfig
from models.tft.base.layers import (
    GLU, GatedResidualNetwork, VariableSelectionNetwork,
    PositionalEmbedding, Time2Vec, LearnableFourierFeatures,
    Sparsemax, DropConnect, ScheduledDropPath, MultiQueryAttention
)

@pytest.fixture
def base_config():
    return TFTConfig(
        raw_time_features_dim=27,  # Match your actual data
        raw_static_features_dim=37,  # Match your actual data
        time_varying_categorical_features_cardinalities=[],  # Match your actual data
        static_categorical_features_cardinalities=[],  # Match your actual data
        num_quantiles=3,
        hidden_size=64,
        lstm_layers=2,
        attention_heads=4,
        dropout_rate=0.1,
        use_positional_encoding=True,
        use_dropconnect=False,
        use_scheduled_drop_path=False,
        drop_path_rate=0.1,
        kernel_initializer="glorot_uniform",
        use_glu_in_grn=True,
        use_layer_norm_in_grn=True,
        use_multi_query_attention=False,
        use_indrnn=False,
        use_logsparse_attention=False,
        sparsity_factor=4,
        use_evidential_regression=False,
        use_mdn=False,
        num_mixtures=5,
        use_time2vec=False,
        time2vec_dim=32,
        use_fourier_features=False,
        num_fourier_features=10,
        use_reformer_attention=False,
        num_buckets=8,
        use_sparsemax=False,
        l1_reg=0.0,
        l2_reg=0.0,
        use_gnn=False,  # Set to True if you are using GNN, False otherwise
        gnn_embedding_dim=None,  # Set to the correct dimension if use_gnn is True
        use_transformer=False,  # Set to True if using Transformer, False otherwise
        transformer_embedding_dim=None,   # Set to the correct dimension if use_transformer is True
        seq_len=12  # Usar una longitud de secuencia consistente
    )



def test_tft_output_shape(base_config):
    model = TFT(config=base_config)
    batch_size = 8
    inputs = create_dummy_inputs(base_config, batch_size)
    output = model(inputs)
    assert output.shape == (batch_size, base_config.seq_len, base_config.num_quantiles)

def test_tft_forward_pass_training(base_config):
    model = TFT(config=base_config)
    batch_size = 4
    inputs = create_dummy_inputs(base_config, batch_size)

    # Verificar comportamiento en modo entrenamiento
    output_train = model(inputs, training=True)
    assert output_train.shape == (batch_size,base_config.seq_len, base_config.num_quantiles)


def test_tft_positional_encoding(base_config):
    config = base_config.copy(update={"use_positional_encoding": True})
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)

    attention_weights = model.get_attention_weights(inputs)
    assert attention_weights.shape == (8, config.attention_heads, config.seq_len, config.seq_len)



def test_tft_time2vec(base_config):
    config = base_config.copy(update={"use_time2vec": True, "time2vec_dim": 16})  # Add time2vec_dim
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)
    output = model(inputs)
    assert output.shape == (8, config.seq_len, base_config.num_quantiles)



def test_custom_attention_mechanisms(base_config):
    # Probar MultiQueryAttention
    config = base_config.copy(update={"use_multi_query_attention": True})
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)

    attention_weights = model.get_attention_weights(inputs)
    assert attention_weights is not None
    assert attention_weights.shape == (8, 1, config.seq_len, config.seq_len)  # MQA has 1 head

    # Reset to use standard MultiHeadAttention
    config = base_config.copy(update={"use_multi_query_attention": False})
    model = TFT(config=config)
    attention_weights = model.get_attention_weights(inputs)
    assert attention_weights is not None
    assert attention_weights.shape == (8, config.attention_heads, config.seq_len, config.seq_len)


def test_variable_selection_networks(base_config):
    model = TFT(config=base_config)
    inputs = create_dummy_inputs(base_config, batch_size=8)

    # Desempaquetar los inputs
    if base_config.use_gnn and base_config.use_transformer:
        time_varying_numeric_inputs, time_varying_categorical_inputs, static_numeric_inputs, static_categorical_inputs, time_inputs, gnn_input, transformer_input = inputs
    elif base_config.use_gnn:
        time_varying_numeric_inputs, time_varying_categorical_inputs, static_numeric_inputs, static_categorical_inputs, time_inputs, gnn_input = inputs
        transformer_input = None  # Ensure it's set to None
    elif base_config.use_transformer:
        time_varying_numeric_inputs, time_varying_categorical_inputs, static_numeric_inputs, static_categorical_inputs, time_inputs, transformer_input = inputs
        gnn_input = None
    else:  # Neither GNN nor Transformer
        time_varying_numeric_inputs, time_varying_categorical_inputs, static_numeric_inputs, static_categorical_inputs, time_inputs = inputs
        gnn_input = None  # Ensure these are None
        transformer_input = None

    #  SIEMPRE concatenar time_inputs a time_varying_numeric_inputs
    time_varying_numeric_inputs = tf.concat([time_varying_numeric_inputs, time_inputs], axis=-1)

     # ✅ Desempaquetar correctamente los inputs numéricos ANTES de pasarlos a las VSNs
    static_inputs = tf.split(static_numeric_inputs, num_or_size_splits=base_config.raw_static_features_dim, axis=-1) + []  # Lista vacía para embeddings
    # Usar time_varying_numeric_inputs.shape[-1] DESPUÉS de la concatenación.
    time_varying_inputs = tf.split(time_varying_numeric_inputs, num_or_size_splits=time_varying_numeric_inputs.shape[-1], axis=-1) + []

    static_context, _ = model.vsn_static(static_inputs, training=False) #Usar training = False
    vsn_time_varying_output, _ = model.vsn_time_varying(time_varying_inputs, training=False, context=static_context)
    
    assert static_context is not None
    assert vsn_time_varying_output is not None
    assert static_context.shape == (8, base_config.hidden_size)
    assert vsn_time_varying_output.shape == (8, base_config.seq_len, base_config.hidden_size)


@pytest.fixture
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp("model_save_test")



def test_model_saving_loading(tmpdir, base_config):
    model = TFT(config=base_config)
    inputs = create_dummy_inputs(base_config, batch_size=8)

    # Guardar modelo
    model.save(str(tmpdir) + "/tft_test")

    # Cargar modelo
    loaded_model = TFT()  # No config passed initially
    loaded_model.load(str(tmpdir) + "/tft_test")  # Load config and weights

    # Verificar que la configuración se cargó correctamente
    assert loaded_model.config.raw_time_features_dim == base_config.raw_time_features_dim
    assert loaded_model.config.hidden_size == base_config.hidden_size

    # Realizar una inferencia para verificar que los pesos se cargaron
    output = model(inputs)
    loaded_output = loaded_model(inputs)
    tf.debugging.assert_near(output, loaded_output)  # Compare outputs


#Tests para configuraciones especificas
def test_tft_with_gnn_transformer():
    config = TFTConfig(
        raw_time_features_dim=27,
        raw_static_features_dim=37,
        time_varying_categorical_features_cardinalities=[],
        static_categorical_features_cardinalities=[],
        use_gnn=True,
        gnn_embedding_dim=64,
        use_transformer=True,
        transformer_embedding_dim=768,
        hidden_size=64,  # Example value
        lstm_layers=2,
        dropout_rate=0.1,
        num_quantiles=3,
        attention_heads=4,
        seq_len=20
    )
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)
    output = model(inputs)
    assert output.shape == (8, config.seq_len, config.num_quantiles)


def test_tft_indrnn():
    config = TFTConfig(
        raw_time_features_dim=27,
        raw_static_features_dim=37,
        time_varying_categorical_features_cardinalities=[],
        static_categorical_features_cardinalities=[],
        use_indrnn=True,
        hidden_size=64,  # Example value
        lstm_layers=2,
        dropout_rate=0.1,
        num_quantiles=3,
        attention_heads=4,  # Keep attention heads even with IndRNN
        seq_len = 20
    )
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)
    output = model(inputs)
    assert output.shape == (8, config.seq_len, config.num_quantiles)


def test_tft_mdn():
    config = TFTConfig(
        raw_time_features_dim=27,
        raw_static_features_dim=37,
        time_varying_categorical_features_cardinalities=[],
        static_categorical_features_cardinalities=[],
        use_mdn=True,
        num_mixtures=5,
        hidden_size=64,
        lstm_layers=2,
        dropout_rate=0.1,
        attention_heads=4,
        seq_len=20
    )
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)
    pis, mus, sigmas = model(inputs)
    assert pis.shape == (8, config.seq_len, 1, config.num_mixtures)  # 1 output dim
    assert mus.shape == (8, config.seq_len, 1, config.num_mixtures)
    assert sigmas.shape == (8, config.seq_len, 1, config.num_mixtures)



def test_tft_evidential():
    config = TFTConfig(
        raw_time_features_dim=27,
        raw_static_features_dim=37,
        time_varying_categorical_features_cardinalities=[],
        static_categorical_features_cardinalities=[],
        use_evidential_regression=True,
        hidden_size=64,
        lstm_layers=2,
        dropout_rate=0.1,
        attention_heads=4,
        seq_len=20
    )
    model = TFT(config=config)
    inputs = create_dummy_inputs(config, batch_size=8)
    output = model(inputs)
    assert output.shape == (8, config.seq_len, 4)  # gamma, v, alpha, beta