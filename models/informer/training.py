import tensorflow as tf
from typing import Optional, List, Dict, Callable, Union, Any, Type
from models.informer.config import InformerConfig  # Add this import statement
# from .model import Informer  # Se comenta porque el modelo se pasa como argumento
#Si se usan las funciones de perdida del TFT, importarlas
from models.tft.base.training import quantile_loss, evidential_loss, mdn_loss
#Si se usa el modelo híbrido
from models.tft_gnn.model import TFT_GNN
from models.tft.base.model import TFT
from models.informer.model import Informer
import numpy as np
from core.utils.helpers import load_config
from pipelines.preprocessing.preprocess_data import filter_easy_matches, filter_medium_matches, filter_hard_matches, filter_all_matches
import logging
import datetime

# --- Configurar logging ---  (Opcional, pero recomendado)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Nivel de logging (INFO, DEBUG, etc.)
handler = logging.StreamHandler() #Salida a consola
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@tf.function
def train_one_epoch(model: Union[Informer, TFT_GNN], train_data: tf.data.Dataset, optimizer: tf.keras.optimizers.Optimizer,
                   criterion: Callable, batch_size: int,
                   use_adversarial_training: bool = False,
                   epsilon: float = 0.01,
                   class_weights: Optional[Dict] = None) -> float:
    """Entrena el modelo Informer (o un modelo compatible, como TFT_GNN) por una época.

    Args:
        model: El modelo a entrenar (Informer o TFT_GNN).
        train_data: El conjunto de datos de entrenamiento (tf.data.Dataset).
        optimizer: El optimizador de Keras.
        criterion: La función de pérdida.
        batch_size: El tamaño del lote.
        use_adversarial_training:  Si es True, se usa entrenamiento adversario.
        epsilon: Magnitud de la perturbación adversaria.
        class_weights:  Pesos de clase (opcional).

    Returns:
        La pérdida de entrenamiento promedio para la época.
    """
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
            if isinstance(criterion, list):
                loss = 0
                for c in criterion:
                    loss += c(targets, predictions)
            else:
                loss = criterion(targets, predictions)

            # --- Pesos de Clase (Opcional) ---
            if class_weights:
                # Asumiendo que los targets son enteros y no one-hot
                if len(targets.shape) == 3:
                    sample_weights = tf.gather(class_weights, tf.cast(targets[:,:,0], tf.int32))
                else:
                    sample_weights = tf.gather(class_weights, tf.cast(targets, tf.int32))
                loss = tf.reduce_mean(loss * sample_weights)

            # --- Regularización L1/L2 (Opcional) ---
            if model.losses:  #  Las capas ya se encargan de esto, si se configuró
                loss += tf.add_n(model.losses)

            # --- Entrenamiento Adversario (FGSM - Opcional) ---
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

        # --- Logging (Opcional - Podrías usar tf.summary aquí) ---
        # if batch % 100 == 0:  # Cada 100 batches (ajusta la frecuencia)
        #     logger.info(f"  Batch {batch}, Loss: {loss.numpy():.4f}")

    return total_loss / num_batches


@tf.function
def evaluate_epoch(model: Union[Informer, TFT_GNN], val_data: tf.data.Dataset,
                   criterion: Callable, batch_size: int) -> float:
    """Evalúa el modelo Informer (o un modelo compatible) por una época."""
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


def train_informer(model: Informer,
                   train_data: tf.data.Dataset,
                   val_data: tf.data.Dataset,
                   config: Union[str, Dict, Type["InformerConfig"]],
                   callbacks: Optional[List[Any]] = None,
                   verbose: int = 1):
    """Entrena el modelo Informer."""

    # --- Cargar Configuración (si es necesario) ---
    if isinstance(config, str):
        config = load_config(config)
    training_config = config['training_params']
    model_config = config['model_params']

    # --- Optimizador ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=training_config["learning_rate"])

    # --- Función de Pérdida ---
    if model_config.get("use_evidential_regression"):
        criterion = evidential_loss  # Asume que tienes esta función definida
    elif model_config.get("use_mdn"):
        criterion = mdn_loss  # Asume que tienes esta función definida
    #El informer no usa cuantiles
    # elif model_config.get("num_quantiles") is not None and model_config.get("num_quantiles") > 0:
    #     criterion = lambda y_true, y_pred: quantile_loss(y_true, y_pred, model_config['num_quantiles'])
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
                    filter_func = filter_easy_matches
                elif stage['filter_func'] == 'medium':
                    filter_func = filter_medium_matches
                elif stage['filter_func'] == 'hard':
                    filter_func = filter_hard_matches
                elif stage['filter_func'] == 'all':
                    filter_func = filter_all_matches
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