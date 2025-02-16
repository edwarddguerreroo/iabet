# models/transformer/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from transformers import TFDistilBertModel, DistilBertConfig, DistilBertTokenizerFast
from typing import Dict, Union, Optional
# from .config import TransformerConfig #Si se usa una clase de configuración
from core.utils.helpers import load_config

class TransformerContextual(Model):
    def __init__(self,
                 config: Union[str, Dict] = "config/transformer_config.yaml",
                 model_name: str = 'distilbert-base-uncased',  # Modelo pre-entrenado
                 max_length: int = 128,  # Longitud máxima de la secuencia
                 dropout_rate: float = 0.1,
                 num_classes: Optional[int] = None, #Usar si se hace fine-tuning
                 freeze_transformer: bool = True,  # Congelar el Transformer por defecto
                 **kwargs):
        super(TransformerContextual, self).__init__(**kwargs)

        # --- Cargar Configuración ---
        if isinstance(config, str):
            self.config = load_config(config)['model_params'] #Obtenemos model_params
        else:
            self.config = config
        #Si se cargó bien, se puede usar self.config, sino, se usan los parámetros
        self.model_name = model_name if model_name is not None else self.config.get('model_name', 'distilbert-base-uncased')
        self.max_length = max_length if max_length is not None else self.config.get('max_length', 128)
        self.dropout_rate = dropout_rate if dropout_rate is not None else self.config.get('dropout_rate', 0.1)
        self.num_classes = num_classes if num_classes is not None else self.config.get('num_classes')
        self.freeze_transformer = freeze_transformer if freeze_transformer is not None else self.config.get('freeze_transformer', True)

        # --- Cargar el Tokenizador y el Modelo Pre-entrenado ---
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.transformer = TFDistilBertModel.from_pretrained(self.model_name) #Cargar desde Hugging Face

        # --- Congelar/Descongelar Capas del Transformer ---
        self.transformer.trainable = not self.freeze_transformer

        # --- Capa de Salida (Opcional - para fine-tuning) ---
        self.dropout = Dropout(self.dropout_rate)
        self.classifier = Dense(self.num_classes, activation='softmax') if self.num_classes else None #Para fine tuning


    def call(self, inputs: tf.Tensor, training=None) -> tf.Tensor: #Recibe el tokenizador
        """
        inputs:  Un tensor de strings (batch_size, ) con los textos a procesar.
        """
        # --- Tokenización ---
        encodings = self.tokenizer(
            inputs.numpy().tolist(),  # Convertir el tensor a una lista de Python
            truncation=True,       # Truncar a la longitud máxima
            padding='max_length',        #  Padding
            max_length=self.max_length,
            return_tensors='tf'     # Retornar tensores de TensorFlow
        )
        # --- Pasar por el Transformer ---
        transformer_output = self.transformer(encodings['input_ids'], attention_mask=encodings['attention_mask'], training=training)
        # --- Obtener el Embedding Contextual ---
        # (Usaremos el embedding del token [CLS], que representa la secuencia completa)
        embedding = transformer_output.last_hidden_state[:, 0, :]  # (batch_size, embedding_dim)

        if self.classifier: #Si se hace fine tuning
            embedding = self.dropout(embedding, training=training)
            logits = self.classifier(embedding)
            return logits #Retornar los logits
        else:
            return embedding  # (batch_size, embedding_dim)

    def get_config(self):
        config = super(TransformerContextual, self).get_config()
        config.update({ #Se guardan los parámetros
            'model_name': self.model_name,
            'max_length': self.max_length,
            'dropout_rate': self.dropout_rate,
            'num_classes': self.num_classes,
            'freeze_transformer': self.freeze_transformer
        })
        return config