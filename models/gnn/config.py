# models/gnn/config.py
from pydantic import BaseModel, field_validator, Field
from typing import List, Union, Dict, Optional

class GNNConfig(BaseModel):
    n_hidden: Union[int, List[int]] = Field(..., description="Número de unidades ocultas en cada capa de la GNN.  Puede ser un entero (mismo número para todas las capas) o una lista de enteros (uno por capa).")
    n_classes: int = Field(..., description="Número de clases (si es clasificación) o dimensión del embedding de salida.")
    n_layers: int = Field(2, description="Número de capas de la GNN.", ge=1)
    gnn_type: str = Field("GAT", description="Tipo de GNN ('GCN', 'GAT', 'GraphSAGE').")
    dropout_rate: float = Field(0.1, description="Tasa de dropout.", ge=0.0, le=1.0)
    use_batchnorm: bool = Field(False, description="Usar Batch Normalization.")
    activation: str = Field('elu', description="Función de activación.")
    l1_reg: float = Field(0.0, description="Regularización L1.")
    l2_reg: float = Field(0.001, description="Regularización L2.")

    @field_validator('gnn_type', mode= 'before')
    def gnn_type_must_be_valid(cls, v):
        valid_types = ["GCN", "GAT", "GraphSAGE"]
        if v not in valid_types:
            raise ValueError(f"gnn_type inválido: {v}. Opciones válidas: {valid_types}")
        return v

    @field_validator('n_hidden', mode='before')
    def n_hidden_must_be_valid(cls, v, values):
        if 'n_layers' in values:  #  Asegurarse de que n_layers ya ha sido validado
            if isinstance(v, int):
                if v <= 0:
                    raise ValueError("n_hidden debe ser mayor que cero.")
                #  No es necesario verificar la longitud si es un entero
            elif isinstance(v, list):
                if len(v) != values['n_layers']:
                    raise ValueError(f"Si n_hidden es una lista, debe tener la misma longitud que n_layers ({values['n_layers']})")
                if any(x <= 0 for x in v):
                    raise ValueError("Todos los valores en n_hidden deben ser mayores que cero.")
            else:
                raise ValueError(f"n_hidden debe ser un entero o una lista de enteros")
        return v