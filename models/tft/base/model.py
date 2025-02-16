import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, GRU, LayerNormalization, MultiHeadAttention,
    Concatenate, Embedding, Add, Dropout, Layer, Reshape, TimeDistributed,
    Attention, GlobalAveragePooling1D, Conv1D
)
from tensorflow.keras.models import Model
from typing import List, Dict, Optional, Tuple, Union, Callable
#Importar capas personalizadas
from models.tft.base.layers import DropConnect, ScheduledDropPath, GLU, Time2Vec, Sparsemax, LearnableFourierFeatures, MultiQueryAttention, VariableSelectionNetwork, GatedResidualNetwork, PositionalEmbedding
import tensorflow_probability as tfp  # Para Deep Evidential Regression y MDN
tfd = tfp.distributions
import numpy as np
from .config import TFTConfig #Importar configuración
from core.utils.helpers import load_config
from models.tft.base.indrnn import IndRNN

# --- Deep Evidential Regression Layer ---
class EvidentialRegression(Layer):
    def __init__(self, output_dim: int, **kwargs):
        super(EvidentialRegression, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = Dense(4 * output_dim)  # gamma, v, alpha, beta

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = self.dense(inputs)
        gamma, v, alpha, beta = tf.split(output, 4, axis=-1)

        # Asegurar que los parámetros sean válidos
        gamma = tf.nn.softplus(gamma)  # > 0
        v = tf.nn.softplus(v) + 1e-6  # > 0, Evitar division por 0
        alpha = tf.nn.softplus(alpha) + 1.0  # > 1
        beta = tf.nn.softplus(beta)  # > 0

        return tf.concat([gamma, v, alpha, beta], axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'output_dim': self.output_dim})
        return config

# --- Loss Function para Deep Evidential Regression ---
def evidential_loss(y_true: tf.Tensor, evidential_params: tf.Tensor) -> tf.Tensor:
    gamma, v, alpha, beta = tf.split(evidential_params, 4, axis=-1)
    omega = 2 * beta * (1 + v)

    # Negative Log-Likelihood
    nll = (
        0.5 * tf.math.log(np.pi / v)
        - alpha * tf.math.log(omega)
        + (alpha + 0.5) * tf.math.log(tf.square(y_true - gamma) * v + omega)
        + tf.math.lgamma(alpha)
        - tf.math.lgamma(alpha + 0.5)
    )
    # Regularización (evitar evidencia infinita)
    reg = tf.abs(y_true - gamma) * (2 * v + alpha)
    return tf.reduce_mean(nll + reg)

# --- Mixture Density Network (MDN) Layer ---
class MDNLayer(Layer):
    def __init__(self, output_dim: int, num_mixtures: int, **kwargs):
        super(MDNLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_mixtures = num_mixtures
        self.dense = Dense(output_dim * (2 * num_mixtures + 1))  # Salida tendra: pi, mu, sigma

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = self.dense(inputs)  # (batch_size, seq_len, output_dim * (2 * num_mixtures + 1))
        # Separar los parámetros de la mezcla
        pis, mus, sigmas = tf.split(output, [self.output_dim * self.num_mixtures, self.output_dim * self.num_mixtures,
                                             self.output_dim * self.num_mixtures], axis=-1)

        # Dar forma
        pis = tf.reshape(pis, [-1, tf.shape(inputs)[1], self.output_dim, self.num_mixtures])
        mus = tf.reshape(mus, [-1, tf.shape(inputs)[1], self.output_dim, self.num_mixtures])
        sigmas = tf.reshape(sigmas, [-1, tf.shape(inputs)[1], self.output_dim, self.num_mixtures])

        # Aplicar funciones de activación adecuadas
        pis = tf.nn.softmax(pis)  # (batch_size, seq_len, output_dim, num_mixtures) - Probabilidades de mezcla
        mus = mus  # (batch_size, seq_len, output_dim, num_mixtures) - Medias
        sigmas = tf.nn.softplus(sigmas)  # (batch_size, seq_len, output_dim, num_mixtures) - Desviaciones estándar

        return pis, mus, sigmas
    def get_config(self):
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'num_mixtures': self.num_mixtures
        })
        return config
# --- Loss Function para MDN ---
def mdn_loss(y_true: tf.Tensor, pis: tf.Tensor, mus: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
    """Negative log-likelihood loss para una Mixture Density Network."""
    # Crear una mezcla de distribuciones normales
    mix = tfd.Categorical(probs=pis)
    components = [tfd.Normal(loc=mu, scale=sigma) for mu, sigma in
                  zip(tf.unstack(mus, axis=-1), tf.unstack(sigmas, axis=-1))]
    gmm = tfd.MixtureSameFamily(mixture_distribution=mix,
                                components_distribution=tfd.Independent(tfd.Normal(loc=mus, scale=sigmas),
                                                                        reinterpreted_batch_ndims=1))

    # Calcular la log-probabilidad negativa
    loss = -gmm.log_prob(y_true)  # (batch_size, seq_len)
    return tf.reduce_mean(loss)

class TFT(Model):
    def __init__(self, config: Union[str, Dict, TFTConfig] = "config/tft_config.yaml", **kwargs):
        super(TFT, self).__init__(**kwargs)

        # --- Cargar Configuración ---
        if isinstance(config, str):
            self.config = TFTConfig(**load_config(config)['model_params'])
        elif isinstance(config, dict):
            self.config = TFTConfig(**config)
        else:  # Ya es un objeto TFTConfig
            self.config = config

        # --- Verificaciones de Compatibilidad ---
        if self.config.use_indrnn and self.config.use_dropconnect:
            raise ValueError("IndRNN no es compatible con DropConnect (que es específico de LSTM/GRU).")

        if sum([self.config.use_reformer_attention, self.config.use_logsparse_attention,
                self.config.use_multi_query_attention]) > 1:
            raise ValueError("Solo se puede seleccionar un tipo de atención avanzada a la vez.")

        if self.config.use_evidential_regression and self.config.num_quantiles != 3:
            print("Advertencia: Se ignora num_quantiles ya que se usa regresion evidencial")
        if self.config.use_mdn and self.config.num_quantiles != 3:
            print("Advertencia: Se ignora num_quantiles ya que se usa MDN")

        if self.config.use_mdn and self.config.use_evidential_regression:
            raise ValueError("No se puede usar MDN y regresion evidencial al mismo tiempo")

        if self.config.use_scheduled_drop_path and self.config.dropout_rate == 0.0:
            print("Advertencia: Se usa ScheduledDropPath pero dropout_rate es 0. No tendra efecto")

        # --- Inicialización de Componentes (basada en la configuración) ---

        # --- Embeddings para variables categóricas ---
        self.time_varying_embeddings = [
            Embedding(input_dim=cardinality, output_dim=self.config.hidden_size)
            for cardinality in self.config.time_varying_categorical_features_cardinalities
        ]
        self.static_embeddings = [
            Embedding(input_dim=cardinality, output_dim=self.config.hidden_size)
            for cardinality in self.config.static_categorical_features_cardinalities
        ]

        # --- Variable Selection Networks ---
        num_time_varying_inputs = (
                len(self.config.time_varying_categorical_features_cardinalities)
                + self.config.raw_time_features_dim
        )
        num_static_inputs = (
                len(self.config.static_categorical_features_cardinalities) + self.config.raw_static_features_dim
        )

        self.vsn_time_varying = VariableSelectionNetwork(
            num_inputs=num_time_varying_inputs,
            units=self.config.hidden_size,
            dropout_rate=self.config.dropout_rate,
            context_units=self.config.hidden_size,
            use_glu_in_grn=self.config.use_glu_in_grn,
            l1_reg=self.config.l1_reg,
            l2_reg=self.config.l2_reg
        )
        self.vsn_static = VariableSelectionNetwork(
            num_inputs=num_static_inputs, units=self.config.hidden_size, dropout_rate=self.config.dropout_rate,
            use_glu_in_grn=self.config.use_glu_in_grn,
            l1_reg=self.config.l1_reg,
            l2_reg=self.config.l2_reg
        )

        # --- LSTM Encoder-Decoder o IndRNN Encoder ---
        self.encoder_layers = []
        if self.config.use_indrnn:
            for _ in range(self.config.lstm_layers):
                self.encoder_layers.append(
                    IndRNN(units=self.config.hidden_size, recurrent_initializer="orthogonal", activation="relu")
                )
        else:
            for _ in range(self.config.lstm_layers):
                self.encoder_layers.append(
                    LSTM(
                        self.config.hidden_size,
                        return_sequences=True,
                        return_state=True,
                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg),
                        kernel_initializer=self.config.kernel_initializer,
                    )
                )
        if not self.config.use_indrnn:
            self.decoder_lstm = LSTM(
                self.config.hidden_size,
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.config.l1_reg, l2=self.config.l2_reg),
                kernel_initializer=self.config.kernel_initializer,
            )

        if self.config.use_dropconnect and not self.config.use_indrnn:  # DropConnect solo para LSTM/GRU
            self.dropconnect_encoder_layers = [
                DropConnect(self.config.dropout_rate) for _ in range(self.config.lstm_layers)
            ]
            self.dropconnect_decoder = DropConnect(self.config.dropout_rate)

        # --- Atención ---
        if self.config.use_reformer_attention:
            # self.attention = ReformerAttentionLayer(...)  # Implementar o integrar
            raise NotImplementedError("Reformer Attention no está implementado")
        elif self.config.use_time_kernel_attention:
            raise NotImplementedError("Time Kernel Attention no está implementado")
        elif self.config.use_multi_query_attention:
            self.attention = MultiQueryAttention(d_model=self.config.hidden_size, num_heads=self.config.attention_heads,
                                            dropout_rate=self.config.dropout_rate)
        else:
            self.attention = MultiHeadAttention(
                num_heads=self.config.attention_heads,
                key_dim=self.config.hidden_size,
                dropout=self.config.dropout_rate,
                kernel_initializer=self.config.kernel_initializer,
        )

        self.attention_grn = GatedResidualNetwork(
            self.config.hidden_size, self.config.dropout_rate, use_time_distributed=True,
            kernel_initializer=self.config.kernel_initializer,
            use_glu=self.config.use_glu_in_grn, use_layer_norm=self.config.use_layer_norm_in_grn,
            l1_reg=self.config.l1_reg,
            l2_reg=self.config.l2_reg
        )

        # --- Salida ---
        if self.config.use_evidential_regression:
            self.output_layer = EvidentialRegression(output_dim=1)  # Asumiendo 1 variable objetivo
        elif self.config.use_mdn:
            self.output_layer = MDNLayer(output_dim=1, num_mixtures=self.config.num_mixtures)  # Capa MDN
        else:
            self.output_layer = Dense(self.config.num_quantiles, kernel_initializer=self.config.kernel_initializer)

        # --- Positional Encoding (Opcional) ---
        if self.config.use_positional_encoding:
            self.positional_encoding = PositionalEmbedding(d_model=self.config.hidden_size)

        # GRN para el contexto estático
        self.grn_static_context = GatedResidualNetwork(
            self.config.hidden_size, self.config.dropout_rate, activation="elu", use_glu=self.config.use_glu_in_grn,
            use_layer_norm=self.config.use_layer_norm_in_grn,
            l1_reg=self.config.l1_reg,
            l2_reg=self.config.l2_reg
        )

        # Capa de proyección para la salida del LSTM/IndRNN
        if not self.config.use_indrnn and not self.config.use_reformer_attention:
            self.lstm_projection = Dense(self.config.hidden_size, kernel_initializer=self.config.kernel_initializer)

        # Scheduled Drop Path (Opcional)
        if self.config.use_scheduled_drop_path:
            self.scheduled_drop_path = ScheduledDropPath(self.config.drop_path_rate)

        # --- Capas para GNN y Transformer (si se usan) ---
        if self.config.use_gnn:
            self.gnn_projection = Dense(self.config.hidden_size)  # Proyectar el embedding de la GNN
            self.gnn_context_grn = GatedResidualNetwork(self.config.hidden_size, self.config.dropout_rate, activation="elu", use_time_distributed=False) #No es time distributed
        if self.config.use_transformer:
            self.transformer_projection = Dense(self.config.hidden_size) #Proyectar embedding
            self.transformer_context_grn = GatedResidualNetwork(self.config.hidden_size, self.config.dropout_rate, activation="elu", use_time_distributed=False)


def call(self, inputs: Tuple[tf.Tensor, ...], training=None) -> tf.Tensor:
    time_varying_numeric_inputs, time_varying_categorical_inputs, static_numeric_inputs, static_categorical_inputs, time_inputs = inputs[:5]

    # --- Desempaquetar Entradas ---
    time_varying_numeric_inputs,  # (batch_size, seq_len, num_time_varying_numeric)
    time_varying_categorical_inputs,  # (batch_size, seq_len, num_time_varying_categorical)
    static_numeric_inputs,  # (batch_size, num_static_numeric)
    static_categorical_inputs,  # (batch_size, num_static_categorical)
    time_inputs  = inputs[:5]  # Los primeros 5 son del TFT
    #Las entradas de la GNN y el transformer son opcionales
    gnn_input = inputs[5] if self.config.use_gnn and len(inputs) > 5 else None
    transformer_input = inputs[6] if self.config.use_transformer and len(inputs) > 6 else None

    # --- Time2Vec (Opcional) ---
    if self.config.use_time2vec:
        time_embedding = self.time2vec(time_inputs)
        time_varying_numeric_inputs = Concatenate(axis=-1)([time_varying_numeric_inputs, time_embedding])

    # --- Fourier (Opcional) ---
    if self.config.use_fourier_features:
        fourier_embedding = self.fourier_features(time_inputs)
        time_varying_numeric_inputs = Concatenate(axis=-1)([time_varying_numeric_inputs, fourier_embedding])

    # --- Embeddings para variables categóricas ---
    time_varying_embedded = [
        self.time_varying_embeddings[i](time_varying_categorical_inputs[:, :, i])
        for i in range(len(self.config.time_varying_categorical_features_cardinalities))
    ]  # List of (batch_size, seq_len, hidden_size)
    static_embedded = [
        self.static_embeddings[i](static_numeric_inputs[:, i])
        for i in range(len(self.config.static_categorical_features_cardinalities))
    ]  # List of (batch_size, hidden_size)

    # --- Concatenar entradas para VSNs ---
    time_varying_inputs = [time_varying_numeric_inputs] + time_varying_embedded
    static_numeric_inputs = inputs[2]  # Assuming static numeric inputs are the third element in the inputs tuple
    static_inputs = [static_numeric_inputs] + static_embedded

    # --- Variable Selection Networks ---
    static_context, _ = self.vsn_static(static_inputs, training=training)
    static_context = self.grn_static_context(static_context, training=training)  # (batch_size, hidden_size)

    # --- Procesar Contexto Estático (GNN/Transformer) ---
    #Se unen los embeddings de la GNN y el Transformer (si existen)
    if self.config.use_gnn and gnn_input is not None:
        gnn_embedding = self.gnn_projection(gnn_input)  # (batch_size, gnn_embedding_dim) -> (batch_size, d_model)
        gnn_embedding = self.gnn_context_grn(gnn_embedding, training=training)
        static_context = tf.concat([static_context, gnn_embedding], axis=-1)  # Unir al contexto estatico

    if self.config.use_transformer and transformer_input is not None:
        transformer_embedding = self.transformer_projection(transformer_input)
        transformer_embedding = self.transformer_context_grn(transformer_embedding, training=training)
        static_context = tf.concat([static_context, transformer_embedding], axis=-1)  # Unir al contexto estatico

    # --- Capa Densa para Unificar Contexto (Si es Necesario) ---
    if static_context is not None:
        static_context = Dense(self.config.hidden_size)(static_context) #Proyeccion final
    vsn_time_varying_output, _ = self.vsn_time_varying(time_varying_inputs, training=training, context=static_context)

    # --- Positional Encoding (Opcional) ---
    if self.config.use_positional_encoding:
        vsn_time_varying_output = self.positional_encoding(vsn_time_varying_output)

    # --- LSTM Encoder-Decoder o IndRNN Encoder ---
    encoder_output = vsn_time_varying_output
    initial_state = None

    if self.config.use_indrnn:
        for layer in self.encoder_layers:
            encoder_output, initial_state = layer(encoder_output, initial_state=initial_state)
            initial_state = [initial_state]
    else:
        for i in range(self.config.lstm_layers):
            encoder_output, state_h, state_c = self.encoder_lstm_layers[i](
                encoder_output, training=training, initial_state=initial_state
            )
            if self.config.use_dropconnect:
                encoder_output = self.dropconnect_encoder_layers[i](encoder_output, training=training)
            initial_state = [state_h, state_c]

        decoder_output = self.decoder_lstm(encoder_output, initial_state=initial_state, training=training)
        if self.config.use_dropconnect:
            decoder_output = self.dropconnect_decoder(decoder_output, training=training)

    # --- Atención ---
    if self.config.use_indrnn:
        attention_input = encoder_output
    else:
        attention_input = decoder_output

    attention_output, attention_weights = self.attention(
        q=attention_input,
        v=attention_input,
        k=attention_input,
        training=training,
    )

    attention_output = self.attention_grn(
        attention_output, training=training, context=static_context
    )

    # --- Capa de Proyección y Salida ---
    if not self.config.use_indrnn and not self.config.use_reformer_attention:  # Solo si se usa LSTM y no Reformer
        projected_lstm_output = self.lstm_projection(decoder_output)
        attention_output = Add()([projected_lstm_output, attention_output])

    if self.config.use_scheduled_drop_path:
        attention_output = self.scheduled_drop_path(attention_output, training=training)

    output = self.attention_grn(attention_output, training=training)  # Procesamos de nuevo

    #Seleccion de la capa de salida
    if self.config.use_mdn:
        pis, mus, sigmas = self.output_layer(output)
        return pis, mus, sigmas
    elif self.config.use_evidential_regression:
        output = self.output_layer(output)
        return output
    else:
        output = self.output_layer(output)  # Salida normal
        return output


    def get_attention_weights(self, inputs: Tuple[tf.Tensor, ...]) -> tf.Tensor:
        """
        Función para obtener los pesos de atención (para visualización).
        """
        (
            time_varying_numeric_inputs,
            time_varying_categorical_inputs,
            static_numeric_inputs,
            static_categorical_inputs,
            time_inputs,
        ) = inputs[:5] #Siempre son 5

        # --- Time2Vec (Opcional) ---
        if self.config.use_time2vec:
            time_embedding = self.time2vec(time_inputs)
            time_varying_numeric_inputs = Concatenate(axis=-1)([time_varying_numeric_inputs, time_embedding])

        # --- Fourier (Opcional) ---
        if self.config.use_fourier_features:
            fourier_embedding = self.fourier_features(time_inputs)
            time_varying_numeric_inputs = Concatenate(axis=-1)([time_varying_numeric_inputs, fourier_embedding])


        time_varying_embedded = [
            self.time_varying_embeddings[i](time_varying_categorical_inputs[:, :, i])
            for i in range(len(self.config.time_varying_categorical_features_cardinalities))
        ]
        static_embedded = [
            self.static_embeddings[i](static_categorical_inputs[:, i])
            for i in range(len(self.config.static_categorical_features_cardinalities))
        ]

        time_varying_inputs = [time_varying_numeric_inputs] + time_varying_embedded
        static_inputs = [static_numeric_inputs] + static_embedded

        static_context, _ = self.vsn_static(static_inputs, training=False)
        static_context = self.grn_static_context(static_context, training=False)

        # --- Procesar Contexto Estático (GNN/Transformer) ---
        #Se unen los embeddings de la GNN y el Transformer (si existen)
        if self.config.use_gnn and len(inputs) > 5: #Si se usa GNN y hay mas inputs
            gnn_input = inputs[5] #Obtenemos
            gnn_embedding = self.gnn_projection(gnn_input)  # (batch_size, gnn_embedding_dim) -> (batch_size, d_model)
            gnn_embedding = self.gnn_context_grn(gnn_embedding, training=False) #False
            static_context = tf.concat([static_context, gnn_embedding], axis=-1)  # Unir al contexto estatico

        if self.config.use_transformer and len(inputs) > 6: #Si se usa Transformer y hay mas inputs
            transformer_input = inputs[6] #Obtenemos
            transformer_embedding = self.transformer_projection(transformer_input)
            transformer_embedding = self.transformer_context_grn(transformer_embedding, training=False) #False
            static_context = tf.concat([static_context, transformer_embedding], axis=-1)  # Unir al contexto estatico

        if static_context is not None:
            static_context = Dense(self.config.hidden_size)(static_context) #Proyeccion final
        vsn_time_varying_output, _ = self.vsn_time_varying(time_varying_inputs, training=False, context=static_context)

        if self.config.use_positional_encoding:
            vsn_time_varying_output = self.positional_encoding(vsn_time_varying_output)

        encoder_output = vsn_time_varying_output
        initial_state = None

        if self.config.use_indrnn:
            for layer in self.encoder_layers:
                encoder_output, initial_state = layer(encoder_output, initial_state=initial_state)  # Pasar initial_state
                initial_state = [initial_state]
        else:
            for i in range(self.config.lstm_layers):
                encoder_output, state_h, state_c = self.encoder_lstm_layers[i](
                    encoder_output, training=False, initial_state=initial_state  # training=False
                )
                initial_state = [state_h, state_c]
            decoder_output = self.decoder_lstm(encoder_output, initial_state=initial_state, training=False) #training=False

        # --- Atención ---
        if self.config.use_indrnn:
            attention_input = encoder_output
        else:
            attention_input = decoder_output

        _, attention_weights = self.attention(  # Solo pesos de atención
            q=attention_input,
            v=attention_input,
            k=attention_input,
            training=False,  # training=False
        )
        return attention_weights

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
        self.config = TFTConfig(**config_dict)

        # Construir el modelo (usando la configuración cargada)
        # Necesitamos una entrada "dummy" para construir el grafo del modelo
        dummy_inputs = (
            tf.zeros((1, 10, self.config.raw_time_features_dim)),  # (batch_size, seq_len, raw_time_features)
            tf.zeros((1, 10, len(self.config.time_varying_categorical_features_cardinalities))),  # (batch_size, seq_len, num_cat_time_features)
            tf.zeros((1, self.config.raw_static_features_dim)),  # (batch_size, raw_static_features)
            tf.zeros((1, len(self.config.static_categorical_features_cardinalities))),  # (batch_size, num_cat_static_features)
            tf.zeros((1, 10, 1)) #Time input
        )
        #Si se usa GNN y Transformer
        if self.config.use_gnn:
            dummy_inputs = dummy_inputs + (tf.zeros((1, self.config.gnn_embedding_dim)),) #Input para GNN
        if self.config.use_transformer:
            dummy_inputs = dummy_inputs + (tf.zeros((1, self.config.transformer_embedding_dim)),) #Input para Transformer

        _ = self(dummy_inputs)  # Construir el modelo

        # Cargar los pesos del modelo
        self.load_weights(filepath + "_weights.h5")