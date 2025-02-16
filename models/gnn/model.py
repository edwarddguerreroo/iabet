# models/gnn/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from spektral.layers import GCNConv, GATConv, GraphSageConv  #  Importar capas de Spektral
from spektral.layers import GlobalAvgPool #Se importa correctamente
from typing import Tuple, Union, Optional, Callable, List, Dict
from core.utils.helpers import load_config #Para cargar configuración
from models.gnn.config import GNNConfig #Configuración

class GNN(Model):
    def __init__(self,
                 config: Union[str, Dict, GNNConfig] = None, #Configuración
                 **kwargs):
        super(GNN, self).__init__(**kwargs)

        # --- Cargar Configuración ---
        if config is None: #Si no se proporciona, usar valores por defecto
            self.config = GNNConfig(n_hidden=64, n_classes=64) #Valores por defecto
        elif isinstance(config, str):
            self.config = GNNConfig(**load_config(config, 'model_params')['gnn_params']) #Cargar desde YAML
        elif isinstance(config, dict):
            self.config = GNNConfig(**config)  # Crear desde diccionario y validar
        else:  # Ya es un objeto GNNConfig
            self.config = config

        # --- Capas de la GNN ---
        self.gnn_layers = []
        if isinstance(self.config.n_hidden, int): #Si es un entero, se usa el mismo para todas las capas
            n_hidden = [self.config.n_hidden] * self.config.n_layers
        elif len(self.config.n_hidden) != self.config.n_layers: #Si no, se debe especificar el tamaño de cada capa
            raise ValueError("n_hidden debe ser un entero o una lista de longitud n_layers")
        else: #Si es lista
            n_hidden = self.config.n_hidden

        for i in range(self.config.n_layers):
            if self.config.gnn_type == "GCN":
                self.gnn_layers.append(GCNConv(n_hidden[i], activation=self.config.activation,
                                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg)))
            elif self.config.gnn_type == "GAT":
                self.gnn_layers.append(GATConv(n_hidden[i], activation=self.config.activation,
                                               kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg))) #Atencion
            elif self.config.gnn_type == "GraphSAGE":
                self.gnn_layers.append(GraphSageConv(n_hidden[i], activation=self.config.activation,
                                                     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg,
                                                                                                  l2=self.config.l2_reg))) #Sage
            else:
                raise ValueError(f"Tipo de GNN '{self.config.gnn_type}' no soportado.")

            if self.config.use_batchnorm:
                self.gnn_layers.append(tf.keras.layers.BatchNormalization()) #Normalizacion
            if self.config.dropout_rate > 0.0:
                self.gnn_layers.append(Dropout(self.config.dropout_rate))


        # --- Capa de Salida ---
        self.out_layer = Dense(self.config.n_classes)  #  Sin activación (la activación se aplica en la función de pérdida, o en el TFT)


    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None) -> tf.Tensor:
        # inputs:  (node_features, adjacency_matrix)  (Spektral espera esto)
        # node_features: (num_nodes, num_node_features)
        # adjacency_matrix: (num_nodes, num_nodes)

        x, a = inputs  # Desempaquetar

        # --- Pasar por las Capas de la GNN ---
        for layer in self.gnn_layers:
            if isinstance(layer, (GCNConv, GATConv, GraphSageConv)):
                x = layer([x, a], training=training)  # Pasar features y adj
            else:
                x = layer(x, training=training) #Para capas como Dropout o BatchNorm

        # --- Capa de Salida ---
        output = self.out_layer(x)  # (num_nodes, n_classes)

        # --- Global Pooling (si queremos un único embedding por grafo) ---
        # (Si queremos un embedding *por nodo*, no hacemos pooling)
        output = GlobalAvgPool()(x)  # (batch_size, n_classes) - Promedio sobre todos los nodos

        return output

    def get_config(self):
        config = super(GNN, self).get_config()
        config.update({ #Se guardan todos los parámetros
            'n_hidden': self.config.n_hidden,
            'n_classes': self.config.n_classes,
            'n_layers': self.config.n_layers,
            'gnn_type': self.config.gnn_type,
            'dropout_rate': self.config.dropout_rate,
            'use_batchnorm': self.config.use_batchnorm,
            'activation': self.config.activation,
            'l1_reg': self.config.l1_reg,
            'l2_reg': self.config.l2_reg
        })
        return config