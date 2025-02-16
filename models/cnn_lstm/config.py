# models/cnn_lstm/config.py
from pydantic import BaseModel, field_validator, Field
from typing import List, Dict, Optional, Union, Literal

class CNNLSTMConfig(BaseModel):
    cnn_filters: List[int] = Field(..., description="Número de filtros para cada capa convolucional de la CNN.")
    cnn_kernel_sizes: List[int] = Field(..., description="Tamaños de kernel para cada capa convolucional de la CNN.")
    cnn_pool_sizes: List[int] = Field(..., description="Tamaños de pooling para cada capa convolucional (opcional).")
    cnn_dilation_rates: List[int] = Field(default=[1, 2, 4], description="Tasas de dilatación para las capas convolucionales.")
    lstm_units: int = Field(..., description="Número de unidades en la capa LSTM.", gt=0)
    dropout_rate: float = Field(0.1, description="Tasa de dropout.", ge=0.0, le=1.0)
    dense_units: List[int] = Field([], description="Unidades en capas densas después de la LSTM (opcional).")
    activation: str = Field('relu', description="Función de activación para las capas convolucionales y densas.")
    l1_reg: float = Field(0.0, description="Regularización L1.")
    l2_reg: float = Field(0.0, description="Regularización L2.")
    use_batchnorm: bool = Field(False, description="Usar Batch Normalization.")
    use_attention: bool = Field(False, description="Usar atención después de la LSTM.") # Opcion para usar atención
    attention_heads: Optional[int] = Field(None, description="Número de cabezas de atención", gt=0) #Si se usa atención

    @field_validator('cnn_filters', 'cnn_kernel_sizes', 'cnn_pool_sizes','cnn_dilation_rates', mode='before')
    def check_same_length(cls, v, values, **kwargs):
        if 'cnn_filters' in values and len(values['cnn_filters']) != len(v):
            raise ValueError('cnn_filters, cnn_kernel_sizes, cnn_pool_sizes and cnn_dilation_rates debe tener la misma longitud')
        return v
    @field_validator('attention_heads', mode='before')
    def attention_heads_must_be_positive(cls, v, values):
        if 'use_attention' in values and values['use_attention'] and v is None:
          raise ValueError("Si se usa atención, se debe especificar el numero de cabezas")
        if v is not None and v <= 0:
            raise ValueError("attention_heads must be positive if use_attention is True")
        return v