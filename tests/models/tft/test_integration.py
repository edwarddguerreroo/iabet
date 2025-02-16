# tests/models/tft/test_integration.py
import pytest
import tensorflow as tf
import numpy as np
import pandas as pd
# Importar los módulos necesarios
from models.tft.base.model import TFT
from models.tft.base.config import TFTConfig
from models.gnn.model import GNN
from pipelines.preprocessing.preprocess_data import load_and_preprocess_data, create_graph_data
from models.tft_gnn.model import TFT_GNN
from models.transformer.model import TransformerContextual
from core.utils.helpers import load_config
#Funcion para crear un dataframe de prueba
def create_test_dataframe():
    data = {
        'partido_id': ['1', '1', '2', '2'],
        'equipo_local_id': ['A', 'A', 'C', 'C'],
        'equipo_visitante_id': ['B', 'B', 'D', 'D'],
        'clima_temperatura': [20.0, 20.0, 22.0, 22.0],
        'clima_humedad': [60.0, 60.0, 65.0, 65.0],
        'clima_viento': [5.0, 5.0, 7.0, 7.0],
        'clima_precipitacion': [0.0, 0.0, 1.0, 1.0],
        'cuota_local': [2.1, 2.1, 1.9, 1.9],
        'cuota_empate': [3.3, 3.3, 3.4, 3.4],
        'cuota_visitante': [3.5, 3.5, 4.0, 4.0],
        'ranking_fifa_local': [10, 10, 20, 20],
        'ranking_elo_local': [1800.0, 1800.0, 1700.0, 1700.0],
        'goles_favor_historico_local': [2.0, 2.0, 1.5, 1.5],
        'goles_contra_historico_local': [1.0, 1.0, 1.2, 1.2],
        'posesion_promedio_historico_local': [0.55, 0.55, 0.45, 0.45],
        'pases_completados_promedio_historico_local': [400.0, 400.0, 350.0, 350.0],
        'faltas_cometidas_promedio_historico_local': [12.0, 12.0, 14.0, 14.0],
        'tarjetas_amarillas_promedio_historico_local': [2.0, 2.0, 2.5, 2.5],
        'tarjetas_rojas_promedio_historico_local': [0.1, 0.1, 0.2, 0.2],
        'ranking_fifa_visitante': [15, 15, 25, 25],
        'ranking_elo_visitante': [1700.0, 1700.0, 1600.0, 1600.0],
        'goles_favor_historico_visitante': [1.5, 1.5, 1.0, 1.0],
        'goles_contra_historico_visitante': [1.2, 1.2, 1.5, 1.5],
        'posesion_promedio_historico_visitante': [0.48, 0.48, 0.40, 0.40],
        'pases_completados_promedio_historico_visitante': [380.0, 380.0, 300.0, 300.0],
        'faltas_cometidas_promedio_historico_visitante': [13.0, 13.0, 15.0, 15.0],
        'tarjetas_amarillas_promedio_historico_visitante': [2.2, 2.2, 2.8, 2.8],
        'tarjetas_rojas_promedio_historico_visitante': [0.1, 0.1, 0.3, 0.3],
        'date': pd.to_datetime(['2023-01-01 15:00', '2023-01-08 15:00', '2023-01-15 17:00', '2023-01-22 17:00']),
        'id': ['1', '1', '2', '2'],
        'target': [2, 1, 0, 2]  # Ejemplo de variable objetivo (goles en el siguiente periodo)
    }
    return pd.DataFrame(data)

# Crear un DataFrame de prueba
test_df = create_test_dataframe()
#Guardar en data/processed
test_df.to_csv('data/processed/preprocessed_data.csv', index=False)

# --- Test ---
#Debe correr sin usar curriculum, ni ssl, ni adversarial

def train_one_epoch(model, train_data, optimizer, criterion, batch_size):
    train_loss = 0.0
    for inputs, targets in train_data.batch(batch_size):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = criterion(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss += loss.numpy()
    return train_loss / len(train_data)

def test_tft_gnn_transformer_integration():
    # Crear una configuración para el TFT
    config = {
        'model_type': 'TFT_GNN',
        'data_dir': 'data/processed',
        'model_params': {
            'raw_time_features_dim': 5,  # Ajustar
            'raw_static_features_dim': 27,  #  Ajustar
            'time_varying_categorical_features_cardinalities': [],
            'static_categorical_features_cardinalities': [],
            'hidden_size': 16,
            'lstm_layers': 1,
            'attention_heads': 2,
            'use_gnn': True,
            'gnn_embedding_dim': 8,  # Importante para la integración
            'num_quantiles': 3,  # Necesario
            'use_transformer': True,
            'transformer_embedding_dim': 768,
            'use_time2vec': False,
            'use_fourier_features': False,
            'seq_len': 2,
            'dropout_rate': 0.0, #Usamos 0 para simplificar,
            'use_positional_encoding': False,
            'use_dropconnect': False,
            'use_scheduled_drop_path': False,
            'kernel_initializer': 'glorot_uniform',
            'use_glu_in_grn': True,
            'use_layer_norm_in_grn': True,
            'use_multi_query_attention': False,
            'use_indrnn': False,
            'use_logsparse_attention': False,
            'use_evidential_regression': False,
            'use_mdn': False,
            'use_sparsemax': False,
            'l1_reg': 0.0,
            'l2_reg': 0.0
        },
        'gnn_params': {
            "n_hidden": [8, 8],  # Dos capas GAT con 8 unidades ocultas cada una
            "n_classes": 16,  # El GNN Layer retorna un embedding de dimension 16
            "n_layers": 2,
            "dropout_rate": 0.0,
            "gnn_type": "GAT",
            "use_batchnorm": False,
            "activation": 'elu',
            "l2_reg": 0.0
        },
        'training_params': {
            'learning_rate': 0.001,
            'batch_size': 2,
            'epochs': 2,
            'optimizer': 'Adam',
            'loss': 'mse', #Cambiamos a mse
            'use_curriculum_learning': False,
            'curriculum_stages': [],
            'use_self_supervised_pretraining': False,
            'ssl_tasks': ['masking'],
            'ssl_masking_ratio': 0.15,
            'ssl_epochs': 10,
            'ssl_task_weights': {'masking': 1.0},
            'use_adversarial_training': False,
            'epsilon': 0.01,
            'use_imbalance_handling': False,  # Usar manejo de desbalance
            'imbalance_strategy': 'smote'
        },
    'evaluation_params': { #Agregar para que no de error
        'metrics': ['mae', 'mse', 'rmse'],
        'betting_strategies': ['fixed_stake', 'proportional', 'kelly'],
        'initial_bankroll': 1000.0,
        'fixed_stake': 10.0
    },
    'transformer_data_dir': 'data/raw/text_data', #Directorio de los textos
    "project_id": "your_project_id", #Reemplazar
    "dataset_id": "your_dataset_id",
    "transformer_table_id": "your_table_id",
    "text_column_name": "text", #Columna con el texto
    "match_id_column_name": "partido_id",
    'region': 'us-central1',
    'dataflow': {},
     'freq': 'h'
    }

    # Cargar y preprocesar datos (usando la función que ya tienes)
    # Asegúrate de que use_gnn=True y use_transformer=True
    train_data, val_data, test_data = load_and_preprocess_data('data/processed', config=config, use_gnn=True, use_transformer=True)

    # Crear una instancia del modelo híbrido TFT_GNN
    model = TFT_GNN(TFTConfig(**config['model_params']), config['gnn_params'])

    # --- Ejecutar el Modelo (Entrenamiento por un Número Pequeño de Épocas) ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    criterion = tf.keras.losses.MeanSquaredError()

    for epoch in range(2):  # Solo 2 épocas para la prueba
        train_loss = train_one_epoch(model, train_data, optimizer, criterion, batch_size=config['training_params']['batch_size'])
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    # --- Realizar Predicciones ---
    for inputs, targets in test_data.batch(2):
        if isinstance(model, TFT_GNN):
            tft_inputs, gnn_inputs = inputs[:5], inputs[5]  # Separar entradas para TFT y GNN
            #Debemos agregar el sexto input, que corresponde al transformer
            transformer_input = inputs[6]
            predictions = model((tft_inputs, gnn_inputs, transformer_input), training=False)

        else:
            predictions = model(inputs, training=False)
        print("Predicciones (primeros ejemplos):", predictions.numpy()[:5])
        break  # Solo un batch para la prueba