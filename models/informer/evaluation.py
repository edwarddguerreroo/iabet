# models/informer/evaluation.py
import tensorflow as tf
from typing import Dict, List, Union, Tuple, Optional
# from .model import Informer  # Se comenta porque el modelo se pasa como argumento
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
#Si se usa el modelo híbrido
from models.tft_gnn.model import TFT_GNN
from models.informer.model import Informer

def evaluate_model(model: Union[Informer, TFT_GNN],
                   test_data: tf.data.Dataset,
                   batch_size: int,
                   use_cross_validation: bool = False,
                   n_splits: int = 5) -> Dict[str, Union[float, List[float]]]:
    """
    Evalúa el modelo Informer en un conjunto de datos de prueba.

    Args:
        model: Instancia del modelo Informer entrenado.
        test_data: Dataset de prueba (tf.data.Dataset).
        batch_size: Tamaño del lote para la evaluación.
        use_cross_validation:  Usar validación cruzada (TimeSeriesSplit).
        n_splits: Número de divisiones para la validación cruzada.

    Returns:
        Diccionario con las métricas de evaluación.
    """
    #Si se usa validacion cruzada
    if use_cross_validation:
        # --- Validación Cruzada (Series Temporales) ---
        if isinstance(test_data, tf.data.Dataset):
            # Convertir tf.data.Dataset a arrays de NumPy (necesario para TimeSeriesSplit)
            test_data_np = list(test_data.as_numpy_iterator())
            #Debemos obtener los inputs correctos
            if len(test_data.element_spec[0]) == 4:  # (enc_input, dec_input, enc_time, dec_time)
               X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                    np.concatenate([x[0][1] for x in test_data_np], axis=0),
                    np.concatenate([x[0][2] for x in test_data_np], axis=0),
                    np.concatenate([x[0][3] for x in test_data_np], axis=0))
            elif len(test_data.element_spec[0]) == 5:  # (enc_input, dec_input, enc_time, dec_time, gnn)
                X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                     np.concatenate([x[0][1] for x in test_data_np], axis=0),
                     np.concatenate([x[0][2] for x in test_data_np], axis=0),
                     np.concatenate([x[0][3] for x in test_data_np], axis=0),
                     np.concatenate([x[0][4] for x in test_data_np], axis=0))
            elif len(test_data.element_spec[0]) == 6:  # (enc_input, dec_input, enc_time, dec_time, gnn, transformer)
                X = (np.concatenate([x[0][0] for x in test_data_np], axis=0),
                     np.concatenate([x[0][1] for x in test_data_np], axis=0),
                     np.concatenate([x[0][2] for x in test_data_np], axis=0),
                     np.concatenate([x[0][3] for x in test_data_np], axis=0),
                     np.concatenate([x[0][4] for x in test_data_np], axis=0),
                     np.concatenate([x[0][5] for x in test_data_np], axis=0))
            else:
                raise ValueError(f"Formato de datos desconocido, el dataset tiene {len(test_data.element_spec[0])} inputs")
            y = np.concatenate([x[1] for x in test_data_np], axis=0)
        else:
            X, y = test_data

        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics_cv = { #Métricas
            "mae": [],
            "mse": [],
            "rmse": []
        }

        for train_index, test_index in tscv.split(X[0]):
            # Dividir los datos en conjuntos de entrenamiento y prueba para este fold
            if isinstance(X, tuple):
                X_test_fold = tuple(x[test_index] for x in X)
            else:
                X_test_fold = X[test_index]
            y_test_fold = y[test_index]

            # Evaluar el modelo en el fold de prueba
            predictions = model.predict(X_test_fold, batch_size=batch_size)
            predictions = predictions[:, -1, :]  # Último paso de tiempo

            # Calcular métricas
            y_true = y_test_fold[:, -1, :]  # Último paso de tiempo
            metrics_cv["mae"].append(mean_absolute_error(y_true, predictions))
            metrics_cv["mse"].append(mean_squared_error(y_true, predictions))
            metrics_cv["rmse"].append(np.sqrt(metrics_cv["mse"][-1]))

        # Promediar las métricas de los folds
        metrics = {k: np.mean(v) for k, v in metrics_cv.items()}
    else:
        # --- Evaluación Normal (sin validación cruzada) ---
        predictions = model.predict(test_data.batch(batch_size))  # Predicciones
        predictions = predictions[:, -1, :]  # Último paso de tiempo

        y_true = []
        for _, targets in test_data: #Obtener los targets
            y_true.append(targets.numpy())
        y_true = tf.concat(y_true, axis=0)
        y_true = y_true[:, -1, :]

        mse = tf.keras.losses.MeanSquaredError()(y_true, predictions).numpy()
        mae = tf.keras.losses.MeanAbsoluteError()(y_true, predictions).numpy()
        rmse = np.sqrt(mse)
        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
        }
    return metrics