import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from typing import Tuple, List, Optional, Dict, Union, Callable
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from core.utils.helpers import load_config
import logging
from transformers import DistilBertTokenizerFast
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions, GoogleCloudOptions
from google.cloud import bigquery
import json

# Configurar el logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Funciones de Filtrado para Curriculum Learning (EJEMPLOS - Debes Adaptarlas) ---
def filter_easy_matches(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Filtra partidos "fáciles" basados en la diferencia de goles promedio
    en los últimos partidos.
    """
    # Asegúrate de que tienes las características necesarias calculadas
    # (ej: 'home_team_goals_scored_mean_5', 'away_team_goals_scored_mean_5')
    df['goal_diff'] = abs(df['home_team_goals_scored_mean_5'] - df['away_team_goals_scored_mean_5'])  # Cambiar
    easy_matches_df = df[df['goal_diff'] >= threshold]
    return easy_matches_df

def filter_medium_matches(df: pd.DataFrame, lower_threshold: float = 1.0, upper_threshold: float = 2.0) -> pd.DataFrame:
    """Filtra partidos de dificultad "media"."""
    df['goal_diff'] = abs(df['home_team_goals_scored_mean_5'] - df['away_team_goals_scored_mean_5'])#Cambiar
    medium_matches_df = df[(df['goal_diff'] > lower_threshold) & (df['goal_diff'] <= upper_threshold)]
    return medium_matches_df

def filter_hard_matches(df: pd.DataFrame, threshold:float = 1.0) -> pd.DataFrame:
    """Filtra partidos dificiles"""
    df['goal_diff'] = abs(df['home_team_goals_scored_mean_5'] - df['away_team_goals_scored_mean_5'])
    hard_matches_df = df[df['goal_diff'] < threshold]
    return hard_matches_df

def filter_all_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna todos los partidos (sin filtrar)"""
    return df

def create_time_features(df: pd.DataFrame, time_col: str = 'date', freq: str = 'h') -> pd.DataFrame:
    """
    Crea características temporales a partir de una columna de fecha/hora.
    freq: h (hora), d (dia), b (dia habil), w (semana), m (mes)
    """
    df[time_col] = pd.to_datetime(df[time_col])
    if freq == 'h' or freq == 't' or freq == 's':
        df["hour_of_day"] = df[time_col].dt.hour
        if freq == 't' or freq == 's':
            df['minute'] = df[time_col].dt.minute
            if freq == 's':
                df['second'] = df[time_col].dt.second
    df["day_of_week"] = df[time_col].dt.dayofweek
    df["day_of_month"] = df[time_col].dt.day
    df["month_of_year"] = df[time_col].dt.month

    #Opcional: Codificar como one-hot (si no se usa embedding temporal)
    # df = pd.get_dummies(df, columns=['hour_of_day', 'day_of_week', ...])
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea características de ingeniería a partir de los datos en bruto.

    Args:
        df: DataFrame de entrada.

    Returns:
        DataFrame con las nuevas características.
    """
    # Aquí es donde implementarías tus funciones de feature engineering.
    # Por ejemplo:
    # df['momentum_home'] = ...
    # df['xg_contextual'] = ...
    # ...

    return df  # Devolver el DataFrame modificado
def create_sequences(df, seq_length, label_length, out_length, target, time_varying_features, static_features, time_features = None,
                     predict = False):
    time_varying_numeric_seq = []
    static_numeric_seq = []
    targets = []
    time_seq = []

    for id_ in df["id"].unique():
        subset = df[df["id"] == id_]
        static_values = subset[static_features].iloc[0].values

        # Generative Style Decoder: Dividir en input y target
        if not predict: #Si no es prediccion
            for i in range(seq_length, len(subset) - out_length + 1):
                # Input: [0, seq_len]
                time_varying_numeric_seq.append(subset[time_varying_features].iloc[i-seq_length:i].values) #Hasta i (exclusivo)
                static_numeric_seq.append(static_values)  # Repetir para cada ventana

                # Target: Concatenar [0, label_len] + [seq_len, seq_len + out_len]
                target_values = subset[target].iloc[i-seq_length:i-seq_length + label_length].values.tolist()  # Parte conocida
                target_values.extend([0.0] * out_length)  # Rellenar con ceros (placeholders)
                targets.append(target_values)

                if time_features is not None: #Agregar time features
                    time_values = subset[time_features].iloc[i-seq_length:i-seq_length+label_length+out_length].values.tolist()
                    time_seq.append(time_values)


        else: #Si es para predecir, solo necesitamos la ultima ventana
            time_varying_numeric_seq.append(subset[time_varying_features].iloc[-seq_length:].values)  #Ultima ventana
            static_numeric_seq.append(static_values)  # Repetir para cada ventana
            #No necesitamos target, se agrega un valor dummy
            targets.append([0.0] * (label_length + out_length)) #Lista de 0s
            if time_features is not None:
                time_values = subset[time_features].iloc[-seq_length:].values.tolist()
                time_seq.append(time_values)

    time_varying_numeric_seq = np.array(time_varying_numeric_seq)
    static_numeric_seq = np.array(static_numeric_seq)
    targets = np.array(targets)

    # Crear inputs
    if time_features is not None:
        time_seq = np.array(time_seq)
        features = (time_varying_numeric_seq, static_numeric_seq, time_seq)
    else:
        features = (time_varying_numeric_seq, static_numeric_seq)  # Sin time
    return features, targets

def make_dataset(df, seq_length, label_length, out_length, target, time_varying_features, static_features, time_features=None,
                 shuffle=True, predict=False, gnn_data = None, transformer_data=None): #Nuevos argumentos
    """Crea un tf.data.Dataset a partir de un DataFrame."""

    features, targets = create_sequences(df, seq_length, label_length, out_length, target, time_varying_features,
                                          static_features,
                                          time_features, predict)  # Pasamos predict

    # --- Añadir Datos de GNN y Transformer (si están disponibles) ---
    if gnn_data is not None:
        #Añadir a features
        #Debemos asegurar que el numero de ejemplos sea el mismo
        if len(features) == 2: #Si no hay time features
            features = (features[0], features[1], gnn_data)
        elif len(features) == 3:
            features = (features[0], features[1], features[2], gnn_data)
    if transformer_data is not None:
        #Añadir a features
        if len(features) == 2:  # Si no hay time features
            features = (features[0], features[1], transformer_data)
        elif len(features) == 3: #Si hay time features
            features = (features[0], features[1], features[2], transformer_data)
        elif len(features) == 4: #Si hay GNN
            features = (features[0], features[1], features[2], features[3], transformer_data)

    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(targets))
    return dataset

def create_graph_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea los datos para la GNN (matriz de adyacencia y características de los nodos).

    Args:
        df: DataFrame de pandas con los datos de los partidos.

    Returns:
        Una tupla con:
            - node_features:  Matriz de características de los nodos (NumPy array).
            - adj_matrix: Matriz de adyacencia (NumPy array).
    """

    logger.info("Iniciando creación de datos para la GNN...")

    # --- 1. Crear Mapeos (IDs a Índices) ---
    # (Necesitamos mapear los IDs de los partidos y equipos a índices consecutivos)

    partido_ids = df['partido_id'].unique().tolist()
    equipo_ids = df['equipo_local_id'].unique().tolist() + df['equipo_visitante_id'].unique().tolist()
    #Si tenemos jugadores
    # jugador_ids = df['jugador_id'].unique().tolist()
    equipo_ids = list(set(equipo_ids))  # Eliminar duplicados (equipos que juegan en casa y fuera)
    partido_to_index = {id_: i for i, id_ in enumerate(partido_ids)}
    equipo_to_index = {id_: i for i, id_ in enumerate(equipo_ids)}
    #Si tenemos jugadores
    # jugador_to_index = {id_: i for i, id_ in enumerate(jugador_ids)}

    num_partidos = len(partido_ids)
    num_equipos = len(equipo_ids)
    # num_jugadores = len(jugador_ids) # Si tenemos jugadores

    logger.info(f"Número de nodos: {num_partidos + num_equipos} (Partidos: {num_partidos}, Equipos: {num_equipos})")

    # --- 2. Crear la Matriz de Adyacencia ---
    # (Usaremos una matriz densa para este ejemplo, pero podrías usar una matriz dispersa si tienes muchos nodos)
    #Usaremos solo partidos y equipos
    num_nodes = num_partidos + num_equipos  # + num_jugadores si se agregan
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # Conectar partidos con equipos (local y visitante)
    for _, row in df.iterrows():
        partido_index = partido_to_index[row['partido_id']]
        equipo_local_index = equipo_to_index[row['equipo_local_id']]
        equipo_visitante_index = equipo_to_index[row['equipo_visitante_id']]

        adj_matrix[partido_index, equipo_local_index + num_partidos] = 1.0  # Partido -> Equipo Local
        adj_matrix[partido_index, equipo_visitante_index + num_partidos] = 1.0  # Partido -> Equipo Visitante
        #Si usamos el grafo bidireccional
        adj_matrix[equipo_local_index + num_partidos, partido_index] = 1.0  # Equipo Local -> Partido
        adj_matrix[equipo_visitante_index + num_partidos, partido_index] = 1.0 # Equipo Visitante -> Partido

    logger.info(f"Matriz de adyacencia creada. Forma: {adj_matrix.shape}")
    logger.debug(f"Valores de ejemplo en la matriz de adyacencia: {adj_matrix[:5, :5]}")

    # --- 3. Crear la Matriz de Características de los Nodos ---
    # Definir las características que usaremos para cada tipo de nodo.
    #   *IMPORTANTE*:  ¡Debes adaptar esto a tus datos!
    partido_features = ['clima_temperatura', 'clima_humedad', 'clima_viento', 'clima_precipitacion',
                        'cuota_local', 'cuota_empate', 'cuota_visitante']  #  ¡AJUSTAR!
    equipo_features = ['ranking_fifa', 'ranking_elo',
                       'goles_favor_historico', 'goles_contra_historico',
                       'posesion_promedio_historico', 'pases_completados_promedio_historico',
                       'faltas_cometidas_promedio_historico', 'tarjetas_amarillas_promedio_historico',
                       'tarjetas_rojas_promedio_historico']  #  ¡AJUSTAR!

    num_partido_features = len(partido_features)
    num_equipo_features = len(equipo_features)
    num_features = num_partido_features + num_equipo_features  # + num_jugador_features si se agregan
    node_features = np.zeros((num_nodes, num_features), dtype=np.float32)

    for _, row in df.iterrows():
        partido_index = partido_to_index[row['partido_id']]
        equipo_local_index = equipo_to_index[row['equipo_local_id']]
        equipo_visitante_index = equipo_to_index[row['equipo_visitante_id']]

        # Características del nodo Partido
        for i, feature in enumerate(partido_features):
            try:
                node_features[partido_index, i] = row[feature]
            except KeyError:
                node_features[partido_index, i] = 0.0  # Valor por defecto
                logger.warning(f"Característica '{feature}' faltante en partido {row['partido_id']}. Usando 0.0")

        # Características del nodo Equipo Local
        for i, feature in enumerate(equipo_features):
            try:
                #Si son historicos, agregar el prefijo
                if feature.endswith('_historico'):
                    node_features[equipo_local_index + num_partidos, num_partido_features + i] = row[f'{feature}_local']
                else:
                    node_features[equipo_local_index + num_partidos, num_partido_features + i] = row[feature]
            except KeyError:
                node_features[equipo_local_index + num_partidos, num_partido_features + i] = 0.0
                logger.warning(f"Característica '{feature}_local' faltante en equipo {row['equipo_local_id']}.")

        # Características del nodo Equipo Visitante
        for i, feature in enumerate(equipo_features):
            try:
                if feature.endswith('_historico'):
                    node_features[equipo_visitante_index + num_partidos, num_partido_features + i] = row[
                        f'{feature}_visitante']
                else:
                    node_features[equipo_visitante_index + num_partidos, num_partido_features + i] = row[feature]
            except KeyError:
                node_features[equipo_visitante_index + num_partidos, num_partido_features + i] = 0.0
                logger.warning(f"Característica '{feature}_visitante' faltante en equipo {row['equipo_visitante_id']}.")
    #Si se usan jugadores, se agregan
    # --- Características del Nodo Jugador ---
    # for _, row in df.iterrows():
    #       #Obtener jugadores
    #       pass

    logger.info(f"Matriz de características de nodos creada. Forma: {node_features.shape}")