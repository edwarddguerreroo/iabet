# models/informer/config.py
from pydantic import BaseModel, field_validator, Field
from typing import List, Dict, Optional, Union, Literal

class InformerConfig(BaseModel):
    enc_in: int = Field(..., description="Número de características de entrada del encoder.")
    dec_in: int = Field(..., description="Número de características de entrada del decoder.")
    c_out: int = Field(..., description="Número de características de salida.")
    seq_len: int = Field(..., description="Longitud de la secuencia de entrada (encoder).")
    label_len: int = Field(..., description="Longitud de la secuencia de inicio del decoder (parte conocida).")
    out_len: int = Field(..., description="Longitud de la secuencia de salida (predicción).")
    factor: int = Field(5, description="Factor de muestreo para ProbSparseAttention.", gt=0)
    d_model: int = Field(512, description="Dimensión del modelo.", gt=0)
    n_heads: int = Field(8, description="Número de cabezas de atención.", gt=0)
    e_layers: int = Field(3, description="Número de capas del encoder.", ge=1)
    d_layers: int = Field(2, description="Número de capas del decoder.", ge=1)
    d_ff: int = Field(2048, description="Dimensión de la capa feed-forward.", gt=0)
    dropout_rate: float = Field(0.1, description="Tasa de dropout.", ge=0.0, le=1.0)
    activation: str = Field("relu", description="Función de activación.")
    output_attention: bool = Field(False, description="Retornar pesos de atención.")
    distil: bool = Field(True, description="Usar destilación en el encoder.")
    mix: bool = Field(True, description="Usar mezcla en la atención.")
    embed_type: str = Field('fixed', description="Tipo de embedding temporal ('fixed', 'timeF', 'learned').")
    freq: str = Field('h', description="Frecuencia para Time Features ('s', 't', 'h', 'd', 'b', 'w', 'm').")
    embed_positions: bool = Field(True, description="Usar Positional Embedding")
    conv_kernel_size: int = Field(25, description="Tamaño del kernel para la capa de convolución en Distilling")
    #Nuevos parámetros
    use_time2vec: bool = Field(False, description="Usar Time2Vec")
    time2vec_dim: int = Field(16, description="Dimensión del embedding de Time2Vec", gt=0)
    time2vec_activation: str = Field('sin', description="Activación para la parte periódica de Time2Vec")
    use_fourier_features: bool = Field(False, description="Usar Learnable Fourier Features")
    num_fourier_features: int = Field(10, description="Número de Fourier Features", gt=0)
    use_sparsemax: bool = Field(False, description="Usar Sparsemax en lugar de Softmax en la atención")
    use_indrnn: bool = Field(False, description="Usar IndRNN en lugar de LSTM") #Aunque no se use en esta iteración
    use_logsparse_attention: bool = Field(True, description="Usar ProbSparse Attention")  #Por defecto True
    l1_reg: float = Field(0.0, description="Peso de la regularización L1.")
    l2_reg: float = Field(0.0, description="Peso de la regularización L2.")
    use_scheduled_drop_path: bool = Field(False, description="Usar Scheduled Drop Path.")
    drop_path_rate: float = Field(0.1, description="Tasa inicial de Drop Path.", ge=0.0, le=1.0)
    use_gnn: bool = Field(False, description="Usar GNN")
    gnn_embedding_dim: Optional[int] = Field(None, description="Dimensión de la salida de la GNN")
    use_transformer: bool = Field(False, description="Usar Transformer")
    transformer_embedding_dim: Optional[int] = Field(None, description="Dimensión de la salida del Transformer")

    # --- Validadores ---
    @field_validator('embed_type', mode='before')
    def embed_type_must_be_valid(cls, v):
        valid_embed_types = ['fixed', 'timeF', 'learned']
        if v not in valid_embed_types:
            raise ValueError(f"embed_type inválido: {v}. Opciones válidas: {valid_embed_types}")
        return v

    @field_validator('freq', mode='before')
    def freq_must_be_valid(cls, v):
        valid_freqs = ['s', 't', 'h', 'd', 'b', 'w', 'm']
        if v not in valid_freqs:
            raise ValueError(f"freq inválida: {v}. Opciones válidas: {valid_freqs}")
        return v