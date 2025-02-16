# models/transformer/config.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict

class TransformerConfig(BaseModel):
    model_name: str = Field('distilbert-base-uncased', description="Nombre del modelo pre-entrenado de Hugging Face.")
    max_length: int = Field(128, description="Longitud máxima de la secuencia de tokens.", gt=0)
    dropout_rate: float = Field(0.1, description="Tasa de dropout para la capa de clasificación (si se usa).", ge=0.0, le=1.0)
    num_classes: Optional[int] = Field(None, description="Número de clases para fine-tuning (si se usa). Si es None, se devuelve el embedding.")
    freeze_transformer: bool = Field(True, description="Congelar los pesos del Transformer pre-entrenado.")
    project_id: str = Field(..., description="ID del proyecto de Google Cloud.")
    dataset_id: str = Field(..., description="ID del dataset de BigQuery.")
    text_table_id: str = Field(..., description="Nombre de la tabla de BigQuery que contiene los datos de texto originales.")
    text_column_name: str = Field(..., description="Nombre de la columna que contiene el texto.")
    match_id_column_name: str = Field(..., description="Nombre de la columna que contiene el ID del partido.")
    transformer_temp_table: str = Field(..., description="Nombre de la tabla temporal de BigQuery para los datos tokenizados.")
    region: str = Field(..., description="Región de Google Cloud.")
    dataflow: Dict = Field(..., description="Configuración de Dataflow")

    @field_validator('num_classes')
    def num_classes_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("num_classes must be positive if specified.")
        return v