# models/informer/model.py
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Embedding, Dropout, Conv1D, GlobalAveragePooling1D, Layer
)
from tensorflow.keras.models import Model
from typing import List, Dict, Optional, Tuple, Union
from .layers import ProbSparseAttention, Distilling, EncoderLayer, DecoderLayer
from models.tft.base.layers import Time2Vec, LearnableFourierFeatures, GatedResidualNetwork # Importar capas
from models.tft.base.model import PositionalEmbedding  # Importar Positional Embedding
import json  # Importar json
from core.utils.helpers import load_config
from models.informer.config import InformerConfig  # Importar configuración

class Informer(Model):
    def __init__(self, config: Union[str, Dict, InformerConfig] = "config/informer/informer_base.yaml", **kwargs):
        super(Informer, self).__init__(**kwargs)

        # --- Cargar Configuración ---
        if isinstance(config, str):  # Ruta del archivo YAML
            self.config = InformerConfig(**load_config(config)['model_params'])
        elif isinstance(config, dict):
            self.config = InformerConfig(**config)
        else:  # Ya es un objeto InformerConfig
            self.config = config

        # --- Embedding (Entrada del Encoder) ---
        self.enc_embedding = DataEmbedding(c_in=self.config.enc_in, d_model=self.config.d_model, embed_type=self.config.embed_type,
                                           freq=self.config.freq, dropout_rate=self.config.dropout_rate, embed_positions=self.config.embed_positions,
                                           use_time2vec=self.config.use_time2vec, time2vec_dim=self.config.time2vec_dim,
                                           use_fourier_features=self.config.use_fourier_features, num_fourier_features=self.config.num_fourier_features,
                                           time2vec_activation=self.config.time2vec_activation)

        # --- Encoder ---
        self.encoder = Encoder([
            EncoderLayer(
                ProbSparseAttention(self.config.factor, attention_dropout=self.config.dropout_rate,
                                    output_attention=self.config.output_attention, use_sparsemax=self.config.use_sparsemax),
                self.config.d_model,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout_rate,
                activation=self.config.activation,
                l1_reg = self.config.l1_reg, #Pasar regularización
                l2_reg = self.config.l2_reg
            ) for _ in range(self.config.e_layers)
        ], [
            Distilling(conv_kernel_size=self.config.conv_kernel_size, out_channels=self.config.d_model)
            if self.config.distil else None for _ in range(self.config.e_layers - 1)
        ], norm_layer=LayerNormalization(epsilon=1e-6) if self.config.distil else None)

        # --- Embedding (Entrada del Decoder) ---
        self.dec_embedding = DataEmbedding(c_in=self.config.dec_in, d_model=self.config.d_model, embed_type=self.config.embed_type,
                                           freq=self.config.freq, dropout_rate=self.config.dropout_rate, embed_positions= self.config.embed_positions,
                                           use_time2vec=self.config.use_time2vec, time2vec_dim=self.config.time2vec_dim,
                                           use_fourier_features=self.config.use_fourier_features, num_fourier_features=self.config.num_fourier_features,
                                           time2vec_activation=self.config.time2vec_activation)

        # --- Decoder ---
        self.decoder = Decoder([
            DecoderLayer(
                ProbSparseAttention(self.config.factor, attention_dropout=self.config.dropout_rate,
                                    output_attention=False, use_sparsemax=self.config.use_sparsemax),  # Self attention
                ProbSparseAttention(self.config.factor, attention_dropout=self.config.dropout_rate,
                                    output_attention=self.config.output_attention, use_sparsemax=self.config.use_sparsemax),  # Cross attention
                self.config.d_model,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout_rate,
                activation=self.config.activation,
                l1_reg = self.config.l1_reg,
                l2_reg= self.config.l2_reg
            )
            for _ in range(self.config.d_layers)
        ], norm_layer=LayerNormalization(epsilon=1e-6))

        # --- Capa de Salida ---
        self.projection = Dense(self.config.c_out)  # Capa de salida

        # --- Capas para GNN y Transformer (si se usan) ---
        if self.config.use_gnn:
            self.gnn_projection = Dense(self.config.d_model)  # Proyectar el embedding de la GNN
            self.gnn_context_grn = GatedResidualNetwork(self.config.d_model, self.config.dropout_rate, activation="elu", use_time_distributed=False) #No es time distributed
        if self.config.use_transformer:
            self.transformer_projection = Dense(self.config.d_model) #Proyectar embedding
            self.transformer_context_grn = GatedResidualNetwork(self.config.d_model, self.config.dropout_rate, activation="elu", use_time_distributed=False)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
             enc_mask: Optional[tf.Tensor] = None,
             dec_mask: Optional[tf.Tensor] = None,
             cross_mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None) -> tf.Tensor:

        enc_input, dec_input, enc_time, dec_time = inputs[:4]  # Separamos entradas
        #Las entradas de la GNN y el transformer son opcionales
        gnn_input = inputs[4] if self.config.use_gnn and len(inputs) >4 else None
        transformer_input = inputs[5] if self.config.use_transformer and len(inputs) > 5 else None
        # --- Procesar Contexto Estático (GNN/Transformer) ---
        static_context = []
        if self.config.use_gnn and gnn_input is not None:
            gnn_embedding = self.gnn_projection(gnn_input)  # (batch_size, gnn_embedding_dim) -> (batch_size, d_model)
            gnn_embedding = self.gnn_context_grn(gnn_embedding, training=training)
            static_context.append(gnn_embedding)

        if self.config.use_transformer and transformer_input is not None:
            transformer_embedding = self.transformer_projection(transformer_input)
            transformer_embedding = self.transformer_context_grn(transformer_embedding, training=training)
            static_context.append(transformer_embedding)
        # --- Combinar Contexto Estático ---
        if static_context:
            static_context = tf.concat(static_context, axis=-1)  # (batch_size, combined_context_dim)
            static_context = Dense(self.config.d_model)(static_context) #Proyeccion final
        else:
            static_context = None

        # --- Embedding del Encoder ---
        enc_out = self.enc_embedding(enc_input, enc_time)  # (batch_size, seq_len, d_model)

        # --- Encoder ---
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_mask, training=training, context = static_context)  # Pasar por encoder

        # --- Embedding del Decoder ---
        dec_out = self.dec_embedding(dec_input, dec_time)  # (batch_size, label_len + out_len, d_model)

        # --- Decoder ---
        dec_out = self.decoder(dec_out, enc_out, attn_mask=dec_mask, cross_attn_mask=cross_mask,
                               training=training, context = static_context)  # Pasar por decoder

        # --- Proyección Final ---
        dec_out = self.projection(dec_out)  # (batch_size, label_len + out_len, c_out)

        # Retornar solo la parte de la prediccion
        if self.config.output_attention:
            return dec_out[:, -self.config.out_len:, :], attns  # (batch_size, out_len, c_out)
        return dec_out[:, -self.config.out_len:, :]  # (batch_size, out_len, c_out)

    def get_attention_weights(self, inputs: Tuple[tf.Tensor, ...]) -> List[tf.Tensor]:
        """Obtiene los pesos de atención del encoder y decoder"""
        enc_input, dec_input, enc_time, dec_time = inputs[:4]
        #Componentes de la GNN y el Transformer
        gnn_input = inputs[4] if self.config.use_gnn and len(inputs) > 4 else None
        transformer_input = inputs[5] if self.config.use_transformer and len(inputs) > 5 else None

        # --- Procesar Contexto Estático (GNN/Transformer) ---
        static_context = []
        if self.config.use_gnn and gnn_input is not None:
            gnn_embedding = self.gnn_projection(gnn_input)  # (batch_size, gnn_embedding_dim) -> (batch_size, d_model)
            gnn_embedding = self.gnn_context_grn(gnn_embedding, training=False)
            static_context.append(gnn_embedding)

        if self.config.use_transformer and transformer_input is not None:
            transformer_embedding = self.transformer_projection(transformer_input)
            transformer_embedding = self.transformer_context_grn(transformer_embedding, training=False)
            static_context.append(transformer_embedding)
        if static_context:
            static_context = tf.concat(static_context, axis=-1)  # (batch_size, combined_context_dim)
            static_context = Dense(self.config.d_model)(static_context)  # Proyeccion final
        else:
            static_context = None

        enc_out = self.enc_embedding(enc_input, enc_time)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=None, training=False, context = static_context)  # Obtener pesos del encoder
        dec_out = self.dec_embedding(dec_input, dec_time)

        # Para obtener los pesos del decoder, iteramos
        dec_attns = []
        cross_attns = []
        for layer in self.decoder.layers:
            dec_out, dec_attn, cross_attn = layer(dec_out, enc_out, attn_mask=None, cross_attn_mask=None,
                                                   training=False)
            dec_attns.append(dec_attn)
            cross_attns.append(cross_attn)

        return enc_attns, dec_attns, cross_attns

    def save(self, filepath: str):
        """Guarda el modelo y la configuración."""
        # Guardar los pesos del modelo
        self.save_weights(filepath + "_weights.h5")

        # Guardar la configuración (como diccionario)
        config_dict = self.config.dict()
        with open(filepath + "_config.json", "w") as f:
            json.dump(config_dict, f)


    def load(self, filepath: str):
        """Carga el modelo y la configuración."""

        # Cargar la configuración
        with open(filepath + "_config.json", "r") as f:
            config_dict = json.load(f)
        self.config = InformerConfig(**config_dict)

        # Construir el modelo (usando la configuración cargada)
        # Necesitamos una entrada "dummy" para construir el grafo del modelo
        dummy_inputs = (tf.zeros((1, self.config.seq_len, self.config.enc_in)),
                        tf.zeros((1, self.config.label_len + self.config.out_len, self.config.dec_in)),
                        tf.zeros((1, self.config.seq_len, 4)), #4 time features
                        tf.zeros((1, self.config.label_len + self.config.out_len, 4)))
        #Si se usa GNN y Transformer
        if self.config.use_gnn:
            dummy_inputs = dummy_inputs + (tf.zeros((1, self.config.gnn_embedding_dim)),)  # Input para GNN
        if self.config.use_transformer:
            dummy_inputs = dummy_inputs + (
            tf.zeros((1, self.config.transformer_embedding_dim)),)  # Input para Transformer

        _ = self(dummy_inputs)  # Construir el modelo

        # Cargar los pesos del modelo
        self.load_weights(filepath + "_weights.h5")

# --- Data Embedding (Completo) ---
class DataEmbedding(Layer):
    def __init__(self, c_in: int, d_model: int, embed_type: str = 'fixed', freq: str = 'h',
                 dropout_rate: float = 0.1, embed_positions: bool = True,
                 use_time2vec: bool = False, time2vec_dim: int = 16, time2vec_activation:str = 'sin',
                 use_fourier_features: bool = False, num_fourier_features: int = 10, **kwargs):

        super(DataEmbedding, self).__init__(**kwargs)

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model) if embed_positions else None
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='fixed' else None

        self.use_time2vec = use_time2vec
        if self.use_time2vec:
            self.time2vec = Time2Vec(output_dim=time2vec_dim, activation=time2vec_activation)
        #Agregar Fourier
        self.use_fourier_features = use_fourier_features
        if self.use_fourier_features:
            self.fourier_features = LearnableFourierFeatures(num_features=1, output_dim=num_fourier_features) #Por ahora, 1 feature

        self.dropout = Dropout(dropout_rate)
        #Guardamos la dimension de salida
        self.d_model = d_model
        if self.use_time2vec:
            self.d_model += time2vec_dim * 2 #Time2Vec dobla la dimension
        if self.use_fourier_features:
            self.d_model += num_fourier_features

    def call(self, x: tf.Tensor, x_mark: Optional[tf.Tensor] = None) -> tf.Tensor:
        # x: (batch_size, seq_len, c_in)
        # x_mark: (batch_size, seq_len, num_time_features)
        x = self.value_embedding(x)
        if self.position_embedding is not None:
            x = x + self.position_embedding(x)
        if self.temporal_embedding is not None:
            x = x + self.temporal_embedding(x_mark)

        #Agregar Time2Vec
        if self.use_time2vec:
            x = tf.concat([x, self.time2vec(x_mark)], axis=-1)  # x_mark como entrada
        #Agregar Fourier
        if self.use_fourier_features:
            x = tf.concat([x, self.fourier_features(x_mark)], axis=-1)  # x_mark como entrada

        return self.dropout(x)
    def get_config(self):
        config = super(DataEmbedding, self).get_config()
        config.update({
            'c_in': self.value_embedding.c_in,
            'd_model': self.d_model,  # Usar la dimensión calculada
            'embed_type': self.temporal_embedding.embed_type if self.temporal_embedding else 'fixed',
            'freq': self.temporal_embedding.freq if self.temporal_embedding else None,
            'dropout_rate': self.dropout.rate,
            'embed_positions': self.position_embedding is not None,
            'use_time2vec': self.use_time2vec,
            'time2vec_dim': self.time2vec.output_dim if self.use_time2vec else None,
            'time2vec_activation': self.time2vec.activation if self.use_time2vec else None,
            'use_fourier_features': self.use_fourier_features,
            'num_fourier_features': self.fourier_features.output_dim if self.use_fourier_features else None
        })
        return config

# --- Token Embedding (Para los valores de las características) ---
class TokenEmbedding(Layer):
    def __init__(self, c_in: int, d_model: int, **kwargs):
        super(TokenEmbedding, self).__init__(**kwargs)
        self.tokenConv = Conv1D(filters=d_model, kernel_size=3, padding='same', activation='relu')
        self.norm = LayerNormalization(epsilon=1e-6)
        self.c_in = c_in
        self.d_model = d_model

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.tokenConv(x)  # (batch_size, seq_len, d_model)
        x = self.norm(x)
        return x
    def get_config(self):
        config = super(TokenEmbedding, self).get_config()
        config.update({
            'c_in': self.c_in,
            'd_model': self.d_model
        })
        return config

# --- Positional Embedding ---
class PositionalEmbedding(Layer):
    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

        # Crear la matriz de positional encoding
        pe = tf.Variable(self._get_positional_encoding(max_len, d_model), trainable=False)
        self.pe = tf.expand_dims(pe, 0)  # [1, max_len, d_model]

    def _get_positional_encoding(self, max_len: int, d_model: int) -> tf.Tensor:
        positions = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pe = tf.zeros((max_len, d_model))
        pe_sin = tf.sin(positions * div_term)
        pe_cos = tf.cos(positions * div_term)

        pe_final = tf.concat([
            tf.expand_dims(pe_sin, axis=2),
            tf.expand_dims(pe_cos, axis=2),
        ], axis=2)
        return tf.reshape(pe_final, (max_len, d_model))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]  # Sumar el positional encoding
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'max_len': self.max_len,
        })
        return config
# --- Temporal Embedding (Time Features) ---
class TemporalEmbedding(Layer):
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h', **kwargs):
        super(TemporalEmbedding, self).__init__(**kwargs)

        #El paper define frecuencias: minute, hour, day of week, day of month, month
        #Se pueden agregar otras
        freq_map = { #Frecuencias
            's': 60, #Segundo
            't': 60 * 60, #Minuto
            'h': 60 * 60 * 24, #Hora
            'd': 60 * 60 * 24 * 30, #Dia
            'b': 60 * 60 * 24 * 30, #Dia habil (business day)
            'w': 60 * 60 * 24 * 30 * 4, #Semana
            'm': 60 * 60 * 24 * 30 * 12, #Mes
        }
        self.embed_type = embed_type
        self.freq = freq
        self.d_model = d_model

        if embed_type == 'timeF':
            self.minute_embed = Embedding(freq_map['t'], d_model)
            self.hour_embed = Embedding(freq_map['h'], d_model)
            self.weekday_embed = Embedding(7, d_model) #Dias de la semana
            self.day_embed = Embedding(31, d_model) #Dias del mes
            self.month_embed = Embedding(12, d_model) #Meses
        else:  # fixed
            self.embed = Embedding(freq_map[freq], d_model)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch_size, seq_len, num_time_features)
        if self.embed_type == 'timeF':
            # x: (batch_size, seq_len, 4)  -  [hora, dia_semana, dia_mes, mes]
            minute_x = self.minute_embed(tf.cast(x[:, :, 0], dtype=tf.int32))
            hour_x = self.hour_embed(tf.cast(x[:, :, 1], dtype=tf.int32))
            weekday_x = self.weekday_embed(tf.cast(x[:, :, 2], dtype=tf.int32))
            day_x = self.day_embed(tf.cast(x[:, :, 3], dtype=tf.int32))
            month_x = self.month_embed(tf.cast(x[:, :, 4], dtype=tf.int32))
            return minute_x + hour_x + weekday_x + day_x + month_x
        else:  # 'fixed'
            return self.embed(tf.cast(x, dtype=tf.int32)) #Se hace casting a entero
    def get_config(self):
        config = super(TemporalEmbedding, self).get_config()
        config.update({
            'd_model': self.d_model,
            'embed_type': self.embed_type,
            'freq': self.freq
        })
        return config

# --- Encoder (Completo) ---
class Encoder(Layer):
    def __init__(self, attn_layers: List[Layer], conv_layers: Optional[List[Layer]] = None,
                 norm_layer: Optional[Layer] = None, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers if conv_layers is not None else [None] * len(attn_layers)
        self.norm = norm_layer

    def call(self, x: tf.Tensor, attn_mask: Optional[tf.Tensor] = None,
             training: Optional[bool] = None, context: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        # x: (batch_size, input_seq_len, d_model)
        attns = []
        if self.conv_layers[0] is not None: #Si hay capas de convolucion (distilling)
          for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask, training=training, context=context)
                x = conv_layer(x) #Aplicar distilling
                attns.append(attn)
          x, attn = self.attn_layers[-1](x, context=context) #Ultima capa sin distilling
          attns.append(attn)

        else: #Si no, solo capas de atencion
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, training=training, context=context)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    def get_config(self):
        config = super(Encoder, self).get_config()
        layer_configs = []
        for layer in self.attn_layers: #Serializar cada una de las capas
            layer_configs.append(tf.keras.utils.serialize_keras_object(layer))
        config.update({
            'attn_layers': layer_configs,
            'conv_layers': [tf.keras.utils.serialize_keras_object(layer) for layer in self.conv_layers] if self.conv_layers[0] is not None else None,
            'norm_layer': tf.keras.utils.serialize_keras_object(self.norm) if self.norm is not None else None,

        })
        return config

# --- Decoder (Completo) ---
class Decoder(Layer):
    def __init__(self, layers: List[Layer], norm_layer: Optional[Layer] = None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.layers = layers
        self.norm = norm_layer

    def call(self, x: tf.Tensor, cross: tf.Tensor, attn_mask: Optional[tf.Tensor] = None,
             cross_attn_mask: Optional[tf.Tensor] = None, training: Optional[bool] = None,
             context: Optional[tf.Tensor] = None) -> tf.Tensor:
        # x: (batch_size, target_seq_len, d_model)
        # cross: (batch_size, input_seq_len, d_model)  (salida del encoder)
        for layer in self.layers:
            x, _, _ = layer(x, cross, attn_mask=attn_mask, cross_attn_mask=cross_attn_mask, training=training, context=context)

        if self.norm is not None:
            x = self.norm(x)
        return x

    def get_config(self):
        config = super(Decoder, self).get_config()
        layer_configs = []
        # Serializar las capas
        for layer in self.layers:
            layer_configs.append(tf.keras.utils.serialize_keras_object(layer))
        config.update({
            'layers': layer_configs,
            'norm_layer': tf.keras.utils.serialize_keras_object(self.norm) if self.norm is not None else None,
        })
        return config