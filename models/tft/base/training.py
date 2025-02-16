import tensorflow as tf
from typing import Dict, Any, List, Tuple, Callable, Union, Optional
# from .model import TFT  # Se comenta porque ahora se pasan los modelos
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from pipelines.preprocessing.preprocess_data import filter_easy_matches, filter_medium_matches, filter_hard_matches, filter_all_matches
#Si se usan las funciones de perdida del Informer
from models.informer.model import Informer
from models.informer.layers import *
#Si se usa el modelo hibrido
from models.tft_gnn.model import TFT_GNN 
from models.tft.base.model import TFT
import json
from core.utils.helpers import load_config
from models.tft.base.config import TFTConfig  # Import TFTConfig
import datetime #Para el logging
import logging
import tensorflow_probability as tfp
tfd = tfp.distributions

# --- Configurar logging ---
logger = logging.getLogger(__name__) # __name__ es el nombre del módulo actual
#Si queremos, podemos configurar un logger para training y otro para evaluation

@tf.function
def train_one_epoch(model: Union[tf.keras.Model, TFT_GNN, Informer], train_data: tf.data.Dataset, optimizer: tf.keras.optimizers.Optimizer,
                   criterion: Callable, batch_size: int,
                   use_adversarial_training: bool = False,
                   epsilon: float = 0.01,
                   class_weights: Optional[Dict] = None) -> float:
    """Entrena el modelo por una época."""
    total_loss = 0.0
    num_batches = 0

    for batch, (inputs, targets) in enumerate(train_data.batch(batch_size)):
        with tf.GradientTape() as tape:
            if isinstance(model, TFT_GNN):
                # --- Entrenamiento de TFT_GNN ---
                tft_inputs, gnn_inputs = inputs[:5], inputs[5]  # Desempaquetar las entradas
                if len(inputs) > 6:
                    transformer_input = inputs[6]
                    predictions = model((tft_inputs, gnn_inputs, transformer_input), training=True)
                else:
                    predictions = model((tft_inputs, gnn_inputs), training=True) #Pasamos tupla

            elif isinstance(model, (TFT, Informer)): #Entrenar solo TFT o Informer
                predictions = model(inputs, training=True)
            else: #Si es otro modelo
                predictions = model(inputs, training=True)

            # --- Cálculo de la Pérdida ---
            if isinstance(criterion, list):  # Si son varias funciones de perdida (ej: curriculum)
                loss = 0
                for c in criterion:
                    loss += c(targets, predictions)
            else:
                loss = criterion(targets, predictions)

            if class_weights:
                #Asumiendo que los targets son enteros y no one-hot
                if len(targets.shape) == 3:
                    sample_weights = tf.gather(class_weights, tf.cast(targets[:,:,0], tf.int32))
                else:
                    sample_weights = tf.gather(class_weights, tf.cast(targets, tf.int32))

                loss = tf.reduce_mean(loss * sample_weights)  # Ponderar la perdida

            if model.losses:  # L1/L2 regularization
                loss += tf.add_n(model.losses)

            # --- Entrenamiento Adversario (FGSM) ---
            if use_adversarial_training:
                if isinstance(model, TFT_GNN): #Si es el modelo hibrido
                    #Obtener gradientes con respecto a los inputs del TFT
                    gradients = tape.gradient(loss, tft_inputs)
                    #Crear perturbaciones
                    if isinstance(gradients, tuple):
                        perturbations = tuple([tf.sign(g) * epsilon for g in gradients])
                    else:
                        perturbations = tf.sign(gradients) * epsilon

                    if isinstance(tft_inputs, tuple):
                        perturbed_inputs = tuple([inp + pert for inp, pert in zip(tft_inputs, perturbations)])
                    else:
                        perturbed_inputs = tft_inputs + perturbations
                    #Pasar al modelo, y obtener la perdida
                    if len(inputs) > 6: #Si se usa transformer
                        predictions_adv = model((perturbed_inputs, gnn_inputs, inputs[6]), training=True)
                    else:
                         predictions_adv = model((perturbed_inputs, gnn_inputs), training=True) #Pasar perturbado
                    loss_adv = criterion(targets, predictions_adv)
                    if class_weights:
                        loss_adv = tf.reduce_mean(loss_adv * sample_weights)
                    loss = (loss + loss_adv) / 2.0 #Promedio

                elif isinstance(model, (TFT, Informer)): #Si es solo el TFT o Informer
                    gradients = tape.gradient(loss, inputs)  # Gradiente con respecto a la entrada
                    if isinstance(gradients, tuple):
                        perturbations = tuple([tf.sign(g) * epsilon for g in gradients])
                    else:
                        perturbations = tf.sign(gradients) * epsilon

                    if isinstance(inputs, tuple):
                        perturbed_inputs = tuple([inp + pert for inp, pert in zip(inputs, perturbations)])
                    else:
                        perturbed_inputs = inputs + perturbations
                    predictions_adv = model(perturbed_inputs, training=True)
                    loss_adv = criterion(targets, predictions_adv)
                    if class_weights:
                        loss_adv = tf.reduce_mean(loss_adv * sample_weights)
                    loss = (loss + loss_adv) / 2.0
                #Si hay mas modelos, agregar aqui
        # --- Cálculo de Gradientes y Optimización ---
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        total_loss += loss.numpy()
        num_batches += 1
        # --- Registrar métricas en TensorBoard ---
        tf.summary.scalar('batch_loss', loss, step=optimizer.iterations)
        tf.summary.scalar('learning_rate', optimizer.lr, step=optimizer.iterations)
        # if batch % 10 == 0:  # Ya no es necesario imprimir puntos con el logging
        #     print(".", end="")
    # print("")
    return total_loss / num_batches

@tf.function
def evaluate_epoch(model: Union[tf.keras.Model, TFT_GNN, Informer], val_data: tf.data.Dataset,
                   criterion: Callable, batch_size: int) -> float:
    """Evalúa el modelo (TFT, Informer o TFT_GNN) en una época."""
    total_loss = 0.0
    num_batches = 0

    for batch, (inputs, targets) in enumerate(val_data.batch(batch_size)):
        if isinstance(model, TFT_GNN):
            # --- Evaluación de TFT_GNN ---
            tft_inputs, gnn_inputs = inputs[:5], inputs[5]  # Desempaquetar las entradas
            if len(inputs) > 6:
                transformer_input = inputs[6]
                predictions = model((tft_inputs, gnn_inputs, transformer_input), training=False)
            else:
                predictions = model((tft_inputs, gnn_inputs), training=False)  # Pasamos tupla

        elif isinstance(model, (TFT, Informer)): #Evaluar TFT o Informer
            predictions = model(inputs, training=False)
        else: #Si es otro modelo
             predictions = model(inputs, training=False)

        if isinstance(criterion, list):  # Si son varias funciones de perdida (ej: curriculum)
            loss = 0
            for c in criterion:
                loss += c(targets, predictions)
        else:
            loss = criterion(targets, predictions)
        if model.losses:
            loss += tf.add_n(model.losses)
        total_loss += loss.numpy()
        num_batches += 1

    return total_loss / num_batches


# --- Funciones de Pérdida ---
def quantile_loss(y_true: tf.Tensor, y_pred: tf.Tensor, quantiles: List[float]) -> tf.Tensor:
    """Calcula la pérdida cuantil."""
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, :, i:i+1]
        loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))
        losses.append(loss)
    return tf.reduce_mean(tf.add_n(losses))

def evidential_loss(y_true: tf.Tensor, evidential_params: tf.Tensor) -> tf.Tensor:
    """Pérdida NLL para regresión evidencial (NIG)."""
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

def mdn_loss(y_true: tf.Tensor, pis: tf.Tensor, mus: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
    """Negative log-likelihood loss para una Mixture Density Network."""
    mix = tfp.distributions.Categorical(probs=pis)
    mix = tfp.distributions.Categorical(probs=pis)
    components = [tfd.Normal(loc=mu, scale=sigma) for mu, sigma in
                  zip(tf.unstack(mus, axis=-1), tf.unstack(sigmas, axis=-1))]
    gmm = tfd.MixtureSameFamily(mixture_distribution=mix,
                                components_distribution=tfd.Independent(tfd.Normal(loc=mus, scale=sigmas),
                                                                        reinterpreted_batch_ndims=1))

    # Calcular la log-probabilidad negativa
    loss = -gmm.log_prob(y_true)  # (batch_size, seq_len)
    return tf.reduce_mean(loss)

def pce_loss(y_true, y_pred, quantile_levels):

    # Verificar que 'y_true' tenga la forma correcta (sin la dimensión de cuantiles)
    if len(y_true.shape) == len(y_pred.shape):  # Si son iguales, asumir que se pasó con la dimension extra
       y_true = y_true[:, :, 0] # (batch_size, seq_len)

    losses = []
    for i, q in enumerate(quantile_levels):
        error = y_true - y_pred[:, :, i]  # (batch_size, seq_len)
        # Aplicar la función de pérdida pinball
        loss = tf.where(error >= 0, q * error, (q - 1) * error)
        losses.append(tf.reduce_mean(loss))  # Media sobre todas las muestras y pasos de tiempo

    # Sumar o promediar las pérdidas de todos los cuantiles
    total_loss = tf.reduce_sum(losses)  # O tf.reduce_mean si prefieres promediar

    return total_loss

# --- Función Principal de Entrenamiento ---
def train_tft(model: Union[tf.keras.Model, TFT_GNN, Informer],
              train_data: tf.data.Dataset,
              val_data: tf.data.Dataset,
              config: Union[str, Dict, "TFTConfig"],  # Usamos el objeto config  # Usamos comillas para evitar dependencia circular
              callbacks: Optional[List[Any]] = None,
              verbose: int = 1):
    """Entrena el modelo TFT (o TFT_GNN)."""

    # --- Cargar Configuración (si es necesario) ---
    if isinstance(config, str):
        config = load_config(config)
    training_config = config['training_params']
    model_config = config['model_params']

    # --- Optimizador ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_config["learning_rate"])

    # --- Función de Pérdida ---
    if model_config.get("use_evidential_regression"):
        criterion = evidential_loss
    elif model_config.get("use_mdn"):
        criterion = mdn_loss
    elif model_config.get("num_quantiles") is not None and model_config.get("num_quantiles") > 0:
        criterion = lambda y_true, y_pred: quantile_loss(y_true, y_pred, [0.1, 0.5, 0.9])  # Ajusta los cuantiles
    else:
        criterion = tf.keras.losses.MeanSquaredError()  # O la que sea adecuada por defecto

    # --- Callbacks ---
    if callbacks is None:
        callbacks = []
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks.append(early_stopping)

    # --- Configurar TensorBoard ---
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + config['experiment_name']
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_callback)

    # --- Curriculum Learning ---
    if training_config["use_curriculum_learning"]:
        #Verificar que se hayan definido las funciones de filtrado
        for stage in training_config["curriculum_stages"]:
            if stage["filter_func"] is not None:
                if stage['filter_func'] == 'easy':
                    filter_func = filter_easy_matches  # Asume que has importado la función
                elif stage['filter_func'] == 'medium':
                    filter_func = filter_medium_matches
                elif stage['filter_func'] == 'hard':
                    filter_func = filter_hard_matches
                elif stage['filter_func'] == 'all':  # Nuevo: all
                    filter_func = filter_all_matches  # Se agrega
                else:
                    raise ValueError(f"Función de filtrado desconocida: {stage['filter_func']}")
        #Si se llega aquí, es porque las funciones existen
        for stage in training_config["curriculum_stages"]:
            print(f"Curriculum Learning Stage: {stage['description']}")

            # --- Filtrar Datos ---
            #Debemos filtrar el dataframe, no el dataset
            #Por lo tanto, el curriculum learning se hará ANTES de crear los datasets

            # --- Entrenamiento (una etapa) ---
            model.compile(optimizer=optimizer, loss=criterion)
            history = model.fit(
                train_data.batch(training_config["batch_size"]),  # Usamos el dataset completo, o el filtrado
                validation_data=val_data.batch(training_config["batch_size"]),
                epochs=stage["epochs"],
                callbacks=callbacks,
                verbose=verbose,
                # class_weight=class_weights  #Si se usan pesos
            )
    else:
        # --- Entrenamiento Estándar ---
        model.compile(optimizer=optimizer, loss=criterion)
        history = model.fit(
            train_data.batch(training_config["batch_size"]),
            validation_data=val_data.batch(training_config["batch_size"]),
            epochs=training_config["epochs"],
            callbacks=callbacks,
            verbose=verbose,
            # class_weight=class_weights
        )

    return history