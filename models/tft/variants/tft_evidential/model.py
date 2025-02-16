# models/tft/variants/tft_evidential/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, LayerNormalization, MultiHeadAttention, \
    Concatenate, Embedding, Add, Dropout, Layer, Reshape, TimeDistributed
from models.tft.base.model import TFT
from models.tft.base.layers import EvidentialRegression  # Importar
from models.tft.base.config import TFTConfig

class TFTEvidential(TFT):
    """Variante del TFT con Evidential Regression."""

    def __init__(self, config: TFTConfig, **kwargs):
        super().__init__(config, **kwargs)

        # --- Reemplazar la capa de salida por EvidentialRegression ---
        self.output_layer = EvidentialRegression(output_dim=1)  # Asumiendo una única variable objetivo