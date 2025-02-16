# models/tft/variants/tft_logsparse/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, LayerNormalization, MultiHeadAttention, \
    Concatenate, Embedding, Add, Dropout, Layer, Reshape, TimeDistributed
from models.tft.base.model import TFT
from models.informer.layers import LogSparseAttention  # Importar LogSparseAttention
from models.tft.base.config import TFTConfig

class TFTLogSparse(TFT):
    """Variante del TFT con LogSparse Attention."""

    def __init__(self, config: TFTConfig, **kwargs):
        super().__init__(config, **kwargs)

        # --- Reemplazar la capa de atención por LogSparseAttention ---
        self.attention = LogSparseAttention(d_model=self.config.hidden_size, num_heads=self.config.attention_heads,
                                            dropout_rate=self.config.dropout_rate,
                                            sparsity_factor=self.config.sparsity_factor)