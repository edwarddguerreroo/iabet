# tests/models/gnn/test_model.py
import pytest
import tensorflow as tf
import numpy as np
from models.gnn.model import GNN
from models.gnn.config import GNNConfig  # Import the config class

# Fixture para crear datos de prueba (grafo simple)
@pytest.fixture
def test_data():
    # Características de los nodos (5 nodos, 3 características por nodo)
    node_features = tf.constant([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ], dtype=tf.float32)

    # Matriz de adyacencia (grafo completo)
    adjacency_matrix = tf.constant([
        [0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0]
    ], dtype=tf.float32)

    return node_features, adjacency_matrix

# Fixture para una configuración por defecto
@pytest.fixture
def default_config():
    return GNNConfig(n_hidden=[16, 8], n_classes=2)

# Prueba de creación del modelo (GCN)
def test_gnn_gcn_creation(default_config):
    default_config.gnn_type = "GCN"
    model = GNN(config=default_config)
    assert isinstance(model, GNN)

# Prueba de creación del modelo (GAT)
def test_gnn_gat_creation(default_config):
    default_config.gnn_type = "GAT"
    model = GNN(config=default_config)
    assert isinstance(model, GNN)

# Prueba de creación del modelo (GraphSAGE)
def test_gnn_graphsage_creation(default_config):
    default_config.gnn_type = "GraphSAGE"
    model = GNN(config=default_config)
    assert isinstance(model, GNN)

# Prueba de la forma de la salida
def test_gnn_output_shape(test_data, default_config):
    model = GNN(config=default_config)
    node_features, adjacency_matrix = test_data
    inputs = (node_features, adjacency_matrix)
    output = model(inputs)
    assert output.shape == (1, 2)  # Un solo grafo en el batch, salida de dimensión 2

# Prueba con entrenamiento activado
def test_gnn_forward_pass_training(test_data, default_config):
    default_config.dropout_rate = 0.2
    default_config.use_batchnorm = True
    model = GNN(config=default_config)
    node_features, adjacency_matrix = test_data
    inputs = (node_features, adjacency_matrix)
    output = model(inputs, training=True)
    assert output.shape == (1, 2)

# Prueba con diferentes tipos de GNN (parametrización)
@pytest.mark.parametrize("gnn_type", ["GCN", "GAT", "GraphSAGE"])
def test_gnn_types(test_data, gnn_type, default_config):
    default_config.gnn_type = gnn_type
    model = GNN(config=default_config)
    node_features, adjacency_matrix = test_data
    inputs = (node_features, adjacency_matrix)
    output = model(inputs)
    assert output.shape == (1, 2)

# Prueba con diferentes números de capas (parametrización)
@pytest.mark.parametrize("n_layers, n_hidden", [(1, 8), (2, [16, 8]), (3, [32, 16, 8])])
def test_gnn_layers(test_data, n_layers, n_hidden):
    config = GNNConfig(n_hidden=n_hidden, n_classes=2, n_layers=n_layers) #Usamos una config para este test
    model = GNN(config=config)
    node_features, adjacency_matrix = test_data
    inputs = (node_features, adjacency_matrix)
    output = model(inputs)
    assert output.shape == (1, 2)

# Prueba de get_config
def test_gnn_get_config(default_config):
    model = GNN(config=default_config)
    config = model.get_config()
    assert isinstance(config, dict)
    # Verificar que la configuración contiene los valores correctos
    assert config['n_hidden'] == default_config.n_hidden
    assert config['n_classes'] == default_config.n_classes
    assert config['dropout_rate'] == default_config.dropout_rate
    assert config['use_batchnorm'] == default_config.use_batchnorm
    assert config['activation'] == default_config.activation
    assert config['l2_reg'] == (default_config.l2_reg,)  # Verificar que el valor de l2_reg es correcto