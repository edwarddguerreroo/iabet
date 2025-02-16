import tensorflow as tf
from typing import Dict, List, Union, Tuple, Optional
from models.tft.base.model import TFT  # Importar las clases TFT 
import numpy as np
from sklearn.model_selection import TimeSeriesSplit  # Para validación cruzada
from sklearn.metrics import brier_score_loss, log_loss  # Para clasificación
from models.informer.model import Informer

def evaluate_model(model: tf.keras.Model,
                   test_data: tf.data.Dataset,
                   batch_size: int,
                   quantiles: Optional[List[float]] = None,
                   use_cross_validation: bool = False,
                   n_splits: int = 5) -> Dict[str, Union[float, List[float]]]:
    """
    Evalúa el modelo TFT (o un modelo compatible) en un conjunto de datos de prueba.

    Args:
        model: Instancia del modelo entrenado.
        test_data: Dataset de prueba (tf.data.Dataset o tuplas de arrays NumPy).
        batch_size: Tamaño del lote para la evaluación.
        quantiles: Lista de cuantiles utilizados durante el entrenamiento (si aplica).
        use_cross_validation:  Usar validación cruzada (TimeSeriesSplit).
        n_splits: Número de divisiones para la validación cruzada.

    Returns:
        Diccionario con las métricas de evaluación.
    """

    if use_cross_validation:
        # --- Validación Cruzada (Series Temporales) ---
        if isinstance(test_data, tf.data.Dataset):
            # Convertir tf.data.Dataset a arrays de NumPy (necesario para TimeSeriesSplit)
            test_data_np = list(test_data.as_numpy_iterator())
            #Debemos acceder a los inputs correctos
            if len(test_data.element_spec[0]) == 3:  # (time_varying, static, time)
                X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                    np.concatenate([x[0][1] for x in test_data_np], axis=0),
                    np.concatenate([x[0][2] for x in test_data_np], axis=0))

            elif len(test_data.element_spec[0]) == 2:  # (time_varying, static)
                X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                    np.concatenate([x[0][1] for x in test_data_np], axis=0))
            elif len(test_data.element_spec[0]) == 4:  # (time_varying, static, time, gnn)
                X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                    np.concatenate([x[0][1] for x in test_data_np], axis=0),
                    np.concatenate([x[0][2] for x in test_data_np], axis=0),
                    np.concatenate([x[0][3] for x in test_data_np], axis=0))
            elif len(test_data.element_spec[0]) == 5:  # (time_varying, static, time, gnn, transformer)
                X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                    np.concatenate([x[0][1] for x in test_data_np], axis=0),
                    np.concatenate([x[0][2] for x in test_data_np], axis=0),
                    np.concatenate([x[0][3] for x in test_data_np], axis=0),
                    np.concatenate([x[0][4] for x in test_data_np], axis=0))
            else:
                raise ValueError(f'Formato de datos desconocido, el dataset contiene {len(test_data.element_spec[0])} inputs.')
            y = np.concatenate([x[1] for x in test_data_np], axis=0) #Los targets
        else:  # Si ya son arrays NumPy (X, y)
            X, y = test_data

        tscv = TimeSeriesSplit(n_splits=n_splits) #Split temporal
        metrics_cv = {  #  Diccionario para almacenar las métricas de cada fold
            "mae": [],
            "mse": [],
            "rmse": [],
            "brier_score": [],  # Para clasificación
            "log_loss": [],    # Para clasificación
            "coverage_90": [] # Cobertura
        }

        #Determinar el tipo de modelo.
        if hasattr(model, 'tft'):
            model_type = type(model.tft)
        else:
            model_type = type(model)

        for train_index, test_index in tscv.split(X[0]):  # Itera sobre los folds

            # Dividir los datos en conjuntos de entrenamiento y prueba para este fold
            if isinstance(X, tuple):
                X_test_fold = tuple(x[test_index] for x in X) #Tupla con los datos
            else:
                X_test_fold = X[test_index] #Solo un array
            y_test_fold = y[test_index] #Los targets

            # Evaluar el modelo en el fold de prueba
            predictions = model.predict(X_test_fold, batch_size=batch_size)
            # --- Métricas para Regresión Cuantil ---
            if quantiles is not None and model_type is TFT and not model.config.use_evidential_regression and not model.config.use_mdn:
                y_true = y_test_fold
                if len(y_true.shape) == 2: #Si no tiene la dimensión de cuantiles
                    y_true = tf.expand_dims(y_true, axis=-1)  # (batch_size, seq_len) -> (batch_size, seq_len, 1)

                q50_idx = quantiles.index(0.5)  # Índice del cuantil 0.5 (mediana)
                y_pred_median = predictions[:, :, q50_idx]  # Predicción mediana
                mae = tf.reduce_mean(tf.abs(y_true[:, :, 0] - y_pred_median)).numpy()

                lower_bound = predictions[:, :, 0]  # Límite inferior del intervalo de confianza
                upper_bound = predictions[:, :, -1]  # Límite superior del intervalo de confianza
                coverage = tf.reduce_mean(
                    tf.cast((y_true[:, :, 0] >= lower_bound) & (y_true[:, :, 0] <= upper_bound), dtype=tf.float32)
                ).numpy()

                # --- Brier Score y Log Loss (si es clasificación) ---
                if hasattr(model, 'output_layer') and (model.output_layer.activation == tf.keras.activations.sigmoid or model.output_layer.activation == tf.keras.activations.softmax):
                    y_prob = predictions[:, :, q50_idx] #Probabilidad predicha
                    #Si y_true es (num_ejem, seq_len, 1) y es de enteros -> one-hot
                    if len(y_true.shape) == 3 and y_true.dtype == 'int32':
                        num_classes = model.output_layer.units
                        y_true_one_hot = tf.one_hot(y_true[:, :, 0], depth=num_classes)
                    #Si y_true es (num_ejem, seq_len) y es de enteros
                    elif len(y_true.shape) == 2 and y_true.dtype == 'int32':
                        num_classes = model.output_layer.units  # Obtener de la ultima capa
                        y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
                    #Si se usa one-hot
                    else:
                        y_true_one_hot = y_true

                    #Debemos aplanar
                    y_true_one_hot_flat = y_true_one_hot.numpy().reshape(-1, y_true_one_hot.shape[-1])
                    y_prob_flat = y_prob.reshape(-1, y_prob.shape[-1])

                    brier_score = brier_score_loss(y_true_one_hot_flat, y_prob_flat)
                    log_loss_value = log_loss(y_true_one_hot_flat, y_prob_flat)
                else: #Si no es clasificación
                    brier_score = np.nan
                    log_loss_value = np.nan

                #Guardar las métricas
                metrics_cv["mae"].append(mae)
                metrics_cv['coverage_90'].append(coverage)
                metrics_cv["brier_score"].append(brier_score)
                metrics_cv["log_loss"].append(log_loss_value)
                metrics_cv['mse'].append(0.0) #No aplica
                metrics_cv['rmse'].append(0.0) #No aplica

            #Si se usa MDN
            elif model_type is TFT and model.config.use_mdn:
                #Aquí se deben calcular las metricas para MDN
                metrics_cv['mae'].append(0.0)
                metrics_cv['coverage_90'].append(0.0)
                metrics_cv['brier_score'].append(0.0)
                metrics_cv['log_loss'].append(0.0)
                metrics_cv['mse'].append(0.0)
                metrics_cv['rmse'].append(0.0)
                pass

            #Si se usa Evidential
            elif model_type is TFT and model.config.use_evidential_regression:
                #Aquí se deben calcular las metricas para Evidential
                metrics_cv['mae'].append(0.0)
                metrics_cv['coverage_90'].append(0.0)
                metrics_cv['brier_score'].append(0.0)
                metrics_cv['log_loss'].append(0.0)
                metrics_cv['mse'].append(0.0)
                metrics_cv['rmse'].append(0.0)
                pass

            #Si se usan cuantiles, ya se calcularon
            elif model_type is Informer:
                #Aquí se deben calcular las metricas para Informer
                metrics_cv['mae'].append(0.0)
                metrics_cv['coverage_90'].append(0.0)
                metrics_cv['brier_score'].append(0.0)
                metrics_cv['log_loss'].append(0.0)
                metrics_cv['mse'].append(0.0)
                metrics_cv['rmse'].append(0.0)
                pass
            # --- Métricas para Regresión Estándar ---
            else:
                y_true = y_test_fold
                y_true = y_true[:, -1, :]
                predictions = predictions[:, -1:, :]  # Solo la última predicción
                #Se calculan las metricas por defecto (MSE, RMSE y MAE)
                mse = tf.keras.losses.MeanSquaredError()(y_true, predictions).numpy()
                mae = tf.keras.losses.MeanAbsoluteError()(y_true, predictions).numpy()
                rmse = np.sqrt(mse)
                metrics_cv["mae"].append(mae)
                metrics_cv["mse"].append(mse)
                metrics_cv["rmse"].append(rmse)
                metrics_cv['coverage_90'].append(0.0)  # No aplica
                metrics_cv['brier_score'].append(0.0)
                metrics_cv['log_loss'].append(0.0)


        # Promediar las métricas de los folds
        metrics = {k: np.mean(v) for k, v in metrics_cv.items()}

    else:
        # --- Evaluación Normal (sin validación cruzada) ---
        predictions = model.predict(test_data.batch(batch_size))

        #Debemos convertir el dataset a numpy
        y_true = []
        for _, labels in test_data:
            y_true.append(labels.numpy())
        y_true = tf.concat(y_true, axis=0)

        #Si se usan cuantiles
        if quantiles is not None and model_type is TFT and not model.config.use_evidential_regression and not model.config.use_mdn:
            if len(y_true.shape) == 2:
                y_true = tf.expand_dims(y_true, axis=-1)

            q50_idx = quantiles.index(0.5)
            y_pred_median = predictions[:, :, q50_idx]
            mae = tf.reduce_mean(tf.abs(y_true[:, :, 0] - y_pred_median)).numpy()
            lower_bound = predictions[:, :, 0]
            upper_bound = predictions[:, :, -1]
            coverage = tf.reduce_mean(
                tf.cast((y_true[:, :, 0] >= lower_bound) & (y_true[:, :, 0] <= upper_bound), dtype=tf.float32)).numpy()

            # --- Brier Score y Log Loss (si es clasificación) ---
            if  hasattr(model, 'output_layer') and (model.output_layer.activation == tf.keras.activations.sigmoid or model.output_layer.activation == tf.keras.activations.softmax):
                y_prob = predictions[:, :, q50_idx]  # Probabilidad predicha

                if len(y_true.shape) == 3 and y_true.dtype == 'int32':
                    num_classes = model.output_layer.units
                    y_true_one_hot = tf.one_hot(y_true[:, :, 0], depth=num_classes)

                elif len(y_true.shape) == 2 and y_true.dtype == 'int32':
                    num_classes = model.output_layer.units
                    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)
                else:
                    y_true_one_hot = y_true

                y_true_one_hot_flat = y_true_one_hot.numpy().reshape(-1, y_true_one_hot.shape[-1])
                y_prob_flat = y_prob.reshape(-1, y_prob.shape[-1])
                brier_score = brier_score_loss(y_true_one_hot_flat, y_prob_flat)
                log_loss_value = log_loss(y_true_one_hot_flat, y_prob_flat)
            else:
                brier_score = np.nan
                log_loss_value = np.nan

            metrics = {
                "mae": mae,
                "coverage_90": coverage, #Cobertura
                "brier_score": brier_score,
                "log_loss": log_loss_value,
                'mse': 0.0,  # Placeholder
                'rmse': 0.0  # Placeholder
            }
        #Si es regresión estandar
        elif model_type is TFT and not model.config.use_evidential_regression and not model.config.use_mdn and not model.config.use_multi_query_attention:
            y_true = y_true[:, -1, :]
            predictions = predictions[:, -1:, :]  # Solo la última predicción
            mse = tf.keras.losses.MeanSquaredError()(y_true, predictions).numpy()
            mae = tf.keras.losses.MeanAbsoluteError()(y_true, predictions).numpy()
            rmse = np.sqrt(mse)
            metrics = {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "brier_score": 0.0,  # No aplica
                "log_loss": 0.0,  # No aplica
                "coverage_90": 0.0  # No aplica
            }
        #Si se usa MDN o Evidential, se deben agregar las métricas correspondientes
        else:
            metrics = {}

    return metrics