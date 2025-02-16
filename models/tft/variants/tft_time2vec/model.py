# models/tft/variants/tft_time2vec/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, LayerNormalization, MultiHeadAttention, \
    Concatenate, Embedding, Add, Dropout, Layer, Reshape, TimeDistributed
from models.tft.base.model import TFT
from models.tft.base.layers import Time2Vec  # Importar Time2Vec
from models.tft.base.config import TFTConfig
class TFTTime2Vec(TFT):
    """Variante del TFT con Time2Vec."""

    def __init__(self, config: TFTConfig, **kwargs):
        super().__init__(config, **kwargs)
         # ---  No es necesario, ya que la integración de Time2Vec
         #       se realiza en el método call de la clase base TFT ---