# models/cnn_lstm/model.py
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, LSTM, Dense, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from typing import Dict, Union, List, Optional
from core.utils.helpers import load_config
from models.cnn_lstm.config import CNNLSTMConfig
from models.tft.base.layers import GatedResidualNetwork, MultiHeadAttention #Podemos usar capas del TFT

class CNN_LSTM(Model):
    def __init__(self, config: Union[str, Dict, CNNLSTMConfig] = None, **kwargs):
        """
        Modelo híbrido CNN-LSTM robusto para predicción de series temporales.

        Args:
            config: Configuración del modelo.
            **kwargs: Argumentos adicionales para la clase base (Model).
        """
        super(CNN_LSTM, self).__init__(**kwargs)

        # --- Cargar Configuración ---
        if config is None:  # Si no se proporciona, usar valores por defecto
            self.config = CNNLSTMConfig(cnn_filters=[32, 64], cnn_kernel_sizes=[
                                        3, 3], cnn_pool_sizes=[2, 2], lstm_units=64)  # Valores por defecto
        elif isinstance(config, str):
            self.config = CNNLSTMConfig(
                **load_config(config, 'model_params')['cnn_lstm_params'])  # Cargar desde YAML
        elif isinstance(config, dict):
            self.config = CNNLSTMConfig(**config)  # Crear desde diccionario y validar
        else:  # Ya es un objeto CNNLSTMConfig
            self.config = config

        # --- Capas CNN (Múltiples bloques con Dilatación) ---
        self.cnn_blocks = []
        for i in range(len(self.config.cnn_filters)):  # Usar la longitud de cnn_filters
            filters = self.config.cnn_filters[i]
            kernel_size = self.config.cnn_kernel_sizes[i]
            dilation_rate = self.config.cnn_dilation_rates[i] if hasattr(self.config, 'cnn_dilation_rates') and len(self.config.cnn_dilation_rates) > i else 1
            pool_size = self.config.cnn_pool_sizes[i]

            cnn_block = [
                Conv1D(filters=filters, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate,
                       kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg)),
                Activation(self.config.activation),
                tf.keras.layers.BatchNormalization() if self.config.use_batchnorm else None, #Usar si se configuro
                MaxPool1D(pool_size=pool_size, padding='same') if pool_size > 1 else None,
                Dropout(self.config.dropout_rate) if self.config.dropout_rate > 0.0 else None,
            ]
            #Agregar capas y filtrar Nones
            self.cnn_blocks.append([layer for layer in cnn_block if layer is not None])

        # --- Capa LSTM ---
        self.lstm = LSTM(self.config.lstm_units, return_sequences=self.config.get('use_attention', False),  # Devolver secuencias si se usa atención
                         kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg))
        #Si se usa o no el dropout
        if self.config.dropout_rate > 0.0:
          self.dropout_lstm = Dropout(self.config.dropout_rate)

        # --- Atención (Opcional) ---
        self.use_attention = self.config.get('use_attention', False) #Obtener de configuración
        if self.use_attention:
            self.attention = MultiHeadAttention(num_heads=self.config.get('attention_heads', 4), key_dim=self.config.lstm_units) #Por defecto, 4
            self.attention_grn = GatedResidualNetwork(self.config.lstm_units, self.config.dropout_rate, use_time_distributed=True)

        # --- Capas Densas (Opcionales) ---
        self.dense_layers = []
        for units in self.config.get('dense_units', []): #Usamos get para evitar errores
            self.dense_layers.append(Dense(units, activation=self.config.activation,
                                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg,
                                                                                       l2=self.config.l2_reg)))
            if self.config.use_batchnorm:
                self.dense_layers.append(tf.keras.layers.BatchNormalization()) #Normalizacion
            if self.config.dropout_rate > 0.0:
                self.dense_layers.append(Dropout(self.config.dropout_rate))

        # --- Capa de Salida ---
        self.output_layer = Dense(1)  # Salida escalar (regresión)

    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        # inputs: (batch_size, seq_len, num_features)

        x = inputs

        # --- Capas CNN ---
        for block in self.cnn_blocks:
            for layer in block: #Aplicar cada capa del bloque
                x = layer(x, training=training)

        # --- Capa LSTM ---
        x = self.lstm(x, training=training)  # (batch_size, lstm_units)
        if self.config.dropout_rate > 0.0:
            x = self.dropout_lstm(x, training=training)

        # --- Atención (Opcional) ---
        if self.use_attention:
            attention_output, _ = self.attention(query=x, value=x, key=x, training=training)
            x = self.attention_grn(attention_output, training=training) #Aplicar GRN
            x = tf.squeeze(x, axis=1) #Quitamos la dimension de la secuencia
        # --- Capas Densas ---
        for layer in self.dense_layers:
            x = layer(x, training=training)

        # --- Capa de Salida ---
        output = self.output_layer(x)  # (batch_size, 1)

        return output
    def get_config(self):
        config = super(CNN_LSTM, self).get_config()
        config.update({  # Se guardan *todos* los parámetros de configuración
            'cnn_filters': self.config.cnn_filters,
            'cnn_kernel_sizes': self.config.cnn_kernel_sizes,
            'cnn_pool_sizes': self.config.cnn_pool_sizes,
            'cnn_dilation_rates': self.config.cnn_dilation_rates,
            'lstm_units': self.config.lstm_units,
            'dropout_rate': self.config.dropout_rate,
            'dense_units': self.config.dense_units,
            'activation': self.config.activation,
            'l1_reg': self.config.l1_reg,
            'l2_reg': self.config.l2_reg,
            'use_batchnorm': self.config.use_batchnorm,
            'use_attention': self.config.use_attention
        })
        return config