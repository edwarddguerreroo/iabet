# models/ensemble/model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model
from typing import List, Dict, Union, Tuple
#Importar los modelos
from models.tft.base.model import TFT
from models.informer.model import Informer
from models.tft_gnn.model import TFT_GNN
from models.cnn_lstm.model import CNN_LSTM

class EnsembleModel(Model):
    def __init__(self, base_models: List[tf.keras.Model],
                 meta_model_hidden_units: List[int] = [128, 64],
                 dropout_rate: float = 0.1,
                 output_activation: str = 'linear'): #Tipo de capa de salida

        super(EnsembleModel, self).__init__()
        self.base_models = base_models  # Lista de modelos base
        self.meta_model_hidden_units = meta_model_hidden_units
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation

        # --- Capas del Meta-Modelo ---
        self.meta_model_layers = []
        for units in meta_model_hidden_units:
            self.meta_model_layers.append(Dense(units, activation="relu"))
            if dropout_rate > 0.0:
                self.meta_model_layers.append(Dropout(dropout_rate))
        self.output_layer = Dense(1, activation = output_activation)  #  Ajustar según la tarea

    def call(self, inputs: Union[List[tf.Tensor], Tuple[tf.Tensor, ...]], training=None) -> tf.Tensor:
        # inputs:  Lista/Tupla de datos de entrada para los modelos base

        # --- Obtener Predicciones de los Modelos Base ---
        base_predictions = []
        for i, model in enumerate(self.base_models):
            #Debemos diferenciar que input pasar a cada modelo
            if isinstance(model, (TFT_GNN)):
                #Si es el modelo hibrido
                #Se asume que los datos para la GNN y el Transformer estan al final de la tupla
                model_input = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6])   # Tupla con datos TFT + GNN + Transformer
                pred = model(model_input, training=False)  # Predicción
            elif isinstance(model, (TFT)):
                #Si es solo el TFT
                model_input = inputs[0:5] #Los primeros 5 inputs son para el TFT
                pred = model(model_input, training=False) #Prediccion
            elif isinstance(model, (Informer)):
                #Si es el informer
                model_input = inputs[0:4] #Los inputs del informer
                pred = model(model_input, training=False) #Prediccion
            elif isinstance(model, (CNN_LSTM)):
                #Si es el CNN_LSTM
                model_input = inputs[0] #El input del CNN-LSTM
                pred = model(model_input, training=False)
            else: #Si es otro modelo
                # pred = model(inputs, training=False) #Si el modelo recibe todos los inputs
                raise ValueError(f"Modelo base no reconocido: {type(model)}")

            #Debemos obtener la predicción puntual.
            #Si el modelo tiene varias salidas, seleccionar la adecuada
            if isinstance(pred, tuple):  # Ejemplo: Informer, MDN
                pred = pred[0]  # Tomar la primera salida (predicciones)
            #Si las predicciones son cuantiles, tomar la mediana
            if len(pred.shape) == 3:  # (batch_size, seq_len, num_quantiles)
                pred = pred[:, -1, 1:2]  # Último paso de tiempo, cuantil central (índice 1 si hay 3 cuantiles)

            #Si se usa evidential, obtener el valor medio
            if len(pred.shape) == 3 and pred.shape[-1] == 4:  # Si es (batch_size, seq_len, 4)
                pred = pred[:,:,0] #gamma

            base_predictions.append(pred)

        # --- Combinar Predicciones (Concatenar) ---
        combined_predictions = Concatenate(axis=-1)(base_predictions) # (batch_size, num_models)

        # --- Pasar por el Meta-Modelo ---
        x = combined_predictions
        for layer in self.meta_model_layers:
            x = layer(x, training=training)  # Pasar por las capas ocultas
        output = self.output_layer(x)  # Capa de salida

        return output

    def get_config(self):
        config = super(EnsembleModel, self).get_config()
        config.update({
            'meta_model_hidden_units': self.meta_model_hidden_units,
            'dropout_rate': self.dropout_rate,
            'output_activation': self.output_activation

        })
        return config