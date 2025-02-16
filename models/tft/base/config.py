# models/tft/base/config.py
from pydantic import BaseModel, validator, Field, field_validator
from typing import List, Dict, Optional, Union, Literal

class TFTConfig(BaseModel):
    # --- Model Params ---
    raw_time_features_dim: int = Field(..., description="Dimension of raw time features.")  # ... indica que es obligatorio
    raw_static_features_dim: int = Field(..., description="Dimension of raw static features")
    time_varying_categorical_features_cardinalities: List[int] = Field(...,
        description="Cardinalities of time-varying categorical features.")
    static_categorical_features_cardinalities: List[int] = Field(...,
        description="Cardinalities of static categorical features.")
    num_quantiles: int = Field(3, description="Number of quantiles to predict.", ge=1)  # ge: greater or equal
    hidden_size: int = Field(64, description="Hidden size of the model.", gt=0)  # gt: greater than
    lstm_layers: int = Field(1, description="Number of LSTM layers.", ge=1)
    attention_heads: int = Field(4, description="Number of attention heads.", ge=1)
    dropout_rate: float = Field(0.1, description="Dropout rate.", ge=0.0, le=1.0)
    use_positional_encoding: bool = Field(False, description="Whether to use positional encoding.")
    use_dropconnect: bool = Field(False, description="Whether to use DropConnect (only for LSTM/GRU).")
    use_scheduled_drop_path: bool = Field(False, description="Whether to use Scheduled Drop Path.")
    drop_path_rate: float = Field(0.1, description="Drop path rate.", ge=0.0, le=1.0)
    kernel_initializer: str = Field("glorot_uniform", description="Kernel initializer.")
    use_glu_in_grn: bool = Field(True, description="Whether to use GLU in GRN.")
    use_layer_norm_in_grn: bool = Field(True, description="Whether to use Layer Normalization in GRN.")
    use_multi_query_attention: bool = Field(False, description="Whether to use Multi-Query Attention.")
    use_indrnn: bool = Field(False, description="Whether to use IndRNN.")
    use_logsparse_attention: bool = Field(False, description="Whether to use LogSparse Attention.")
    sparsity_factor: int = Field(2, description="Sparsity factor for LogSparse Attention.", ge=1)
    use_evidential_regression: bool = Field(False, description="Whether to use Evidential Regression.")
    use_mdn: bool = Field(False, description="Whether to use Mixture Density Network.")
    num_mixtures: int = Field(5, description="Number of mixtures for MDN.", ge=1)
    use_time2vec: bool = Field(False, description="Whether to use Time2Vec.")
    time2vec_dim: int = Field(16, description="Dimension of Time2Vec embedding.", gt=0)
    time2vec_activation: str = Field('sin', description="Activation for Time2Vec.")
    use_fourier_features: bool = Field(False, description="Whether to use Learnable Fourier Features.")
    num_fourier_features: int = Field(10, description="Number of Fourier Features", gt=0)
    use_reformer_attention: bool = Field(False, description="Whether to use Reformer Attention.")
    num_buckets: int = Field(8, description="Number of buckets for Reformer/LSH Attention.", ge=1)
    use_sparsemax: bool = Field(False, description="Whether to use Sparsemax.")
    l1_reg: float = Field(0.0, description="L1 regularization weight.")
    l2_reg: float = Field(0.0, description="L2 regularization weight.")
    use_gnn: bool = Field(False, description="Usa GNN")
    gnn_embedding_dim: Optional[int] = Field(None, description="Dimension de la salida de la GNN")
    use_transformer: bool = Field(False, description="Usa Transformer")
    transformer_embedding_dim: Optional[int] = Field(None, description="Dimension de la salida del Transformer")
    seq_len: int = Field(20, description="Longitud de la secuencia") #Se agrega longitud de secuencia

    # --- Validadores ---
    @field_validator("time_varying_categorical_features_cardinalities", "static_categorical_features_cardinalities", mode='before')
    def cardinalities_must_be_positive(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError("Cardinalities must be positive")
        return v