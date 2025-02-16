# models/tft_gnn/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from typing import List, Dict, Optional, Tuple, Union, Callable
from models.tft.base.model import TFT  # Importar la clase base TFT
from models.gnn.model import GNN  # Importar la clase GNN
from models.tft.base.config import TFTConfig #Se importa la configuración
from core.utils.helpers import load_config

class TFT_GNN(Model):
    def __init__(self, tft_config: Union[str, Dict, TFTConfig], gnn_config: Dict, **kwargs):
        """
        Modelo híbrido que combina TFT y GNN.

        Args:
            tft_config: Configuración para el modelo TFT (ruta al YAML, diccionario o objeto TFTConfig).
            gnn_config: Configuración para el modelo GNN (diccionario).
            **kwargs: Argumentos adicionales para la clase base (Model).
        """
        super(TFT_GNN, self).__init__(**kwargs)

        # --- Cargar Configuración del TFT ---
        if isinstance(tft_config, str):
            self.tft_config = TFTConfig(**load_config(tft_config)['model_params'])
        elif isinstance(tft_config, dict):
            self.tft_config = TFTConfig(**tft_config)
        else:  # Ya es un objeto TFTConfig
            self.tft_config = tft_config

        # --- Crear el TFT ---
        self.tft = TFT(config=self.tft_config)

        # --- Crear la GNN ---
        self.gnn = GNN(config=gnn_config)  # Pasar la configuración directamente

        # --- Capa de Combinación ---
        # Usar una capa densa para proyectar la salida de la GNN,
        # y luego la atención para combinar
        self.gnn_projection = Dense(self.tft_config.hidden_size)
        self.attention = tf.keras.layers.Attention() #Capa de atención


    def call(self, inputs: Tuple, training=None) -> tf.Tensor:

        # --- Desempaquetar Entradas ---
        #Asumimos que los inputs del TFT son los primeros 5, luego GNN y por ultimo el Transformer
        tft_inputs, gnn_inputs = inputs[:5], inputs[5]
        transformer_input = inputs[6] if self.tft_config.use_transformer else None #Si se usa el transformer

        # --- Pasar datos por el TFT ---
        tft_output = self.tft(tft_inputs, training=training)

        # --- Pasar datos por la GNN ---
        gnn_output = self.gnn(gnn_inputs, training=training)  # (batch_size, num_nodes, gnn_embedding_dim) o (batch_size, gnn_embedding_dim)
        #Si la salida de la GNN es por nodo, se debe agregar a un solo vector
        if len(gnn_output.shape) == 3:
            #Ejemplo: Promedio de los embeddings de los nodos
            gnn_output = tf.reduce_mean(gnn_output, axis=1) # (batch_size, gnn_embedding_dim)

        # --- Combinar las Salidas ---
        #Usar un mecanismo de atención para combinar las salidas:
        # 1. Proyectar la salida de la GNN
        gnn_output = self.gnn_projection(gnn_output)  # (batch_size, gnn_embedding_dim) -> (batch_size, hidden_size)

        #2. Expandir dimensiones para poder concatenar
        tft_output = tf.expand_dims(tft_output, axis=1)  # (batch_size, 1, hidden_size)
        gnn_output = tf.expand_dims(gnn_output, axis=1)  # (batch_size, 1, hidden_size)

        # 3. Concatenar a lo largo de la dimensión de la secuencia (ahora ambos tienen seq_len=1)
        combined_output = Concatenate(axis=1)([tft_output, gnn_output])  # (batch_size, 2, hidden_size)

        # 4. Aplicar la capa de Atención
        #    Usamos una capa de atención de Keras, que ya incluye el softmax y la multiplicación por los pesos
        attended_output = self.attention([combined_output, combined_output])  # (batch_size, 2, hidden_size)

        # 5. Reducir la dimensión de la secuencia (ej: promedio)
        final_output = tf.reduce_mean(attended_output, axis=1)  # (batch_size, hidden_size)
        return final_output
    def get_attention_weights(self, inputs):
        #Se retorna la atencion del TFT
        tft_inputs, gnn_inputs = inputs[:5], inputs[5]
        return self.tft.get_attention_weights(tft_inputs)