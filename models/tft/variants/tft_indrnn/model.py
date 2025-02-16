# models/tft/variants/tft_indrnn/model.py
import tensorflow as tf
from tensorflow.keras.layers import Layer
from typing import Union, Callable, List, Tuple, Optional
from models.tft.base.model import TFT
from models.tft.base.config import TFTConfig
from models.tft.base.indrnn import IndRNN
class TFTIndRNN(TFT):
    """
    Variante del TFT que utiliza IndRNN en lugar de LSTM en el encoder.
    """

    def __init__(self, config: TFTConfig, **kwargs):
        # Llamar al constructor de la clase base (TFT)
        super().__init__(config, **kwargs)

        # --- Reemplazar las capas LSTM del encoder por IndRNN ---
        self.encoder_layers = []
        for _ in range(self.config.lstm_layers):
            self.encoder_layers.append(
                IndRNN(units=self.config.hidden_size, recurrent_initializer="orthogonal", activation="relu")
            )

        # --- Eliminar las capas LSTM del decoder (ya que IndRNN no las necesita) ---
        delattr(self, "decoder_lstm")
        if hasattr(self, "dropconnect_decoder"):
            delattr(self, "dropconnect_decoder")
        if hasattr(self, "lstm_projection"):
            delattr(self, "lstm_projection")
        #Eliminar dropconnect de encoder si existe
        if hasattr(self, 'dropconnect_encoder_layers'):
            delattr(self,'dropconnect_encoder_layers')

