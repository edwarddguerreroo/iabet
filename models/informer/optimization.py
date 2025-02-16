import optuna
import tensorflow as tf
# from models.informer.model import Informer  #  Importar la clase Informer #Se comenta porque se instancia el modelo base
# from models.informer.training import train_one_epoch, evaluate_epoch  # Importar funciones de entrenamiento/evaluación
from pipelines.preprocessing.preprocess_data import load_and_preprocess_data
from core.utils.helpers import load_config  # Para cargar la configuración
from models.informer.config import InformerConfig
#Si se usa la GNN
from models.tft_gnn.model import TFT_GNN
from models.gnn.model import GNN
from models.informer.model import Informer
#Funciones de perdida y entrenamiento del TFT
from models.tft.base.training import train_one_epoch, evaluate_epoch, quantile_loss, evidential_loss, mdn_loss
import time
import numpy as np
from typing import Tuple

def objective(trial: optuna.trial.Trial, config_path: str = "config/informer/informer_base.yaml") -> Tuple[float, float]:
    """Función objetivo para Optuna (multi-objetivo)."""

    # --- Cargar Configuración Base ---
    config = load_config(config_path)
    base_model_params = config['model_params']
    training_params = config['training_params']

    # --- Hiperparámetros (Espacio de Búsqueda) ---
    model_params = {
        "factor": trial.suggest_categorical("factor", [3, 5, 7]),  # Factor para ProbSparse
        "d_model": trial.suggest_categorical("d_model", [128, 256, 512]),
        "n_heads": trial.suggest_categorical("n_heads", [ 4, 8]),
        "e_layers": trial.suggest_int("e_layers", 1, 4),
        "d_layers": trial.suggest_int("d_layers", 1, 3),
        "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "conv_kernel_size": trial.suggest_categorical("conv_kernel_size", [21, 23, 25, 27, 29]), #Parametros de la capa de distilling
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "output_attention": trial.suggest_categorical('output_attention', [True, False]),
        "distil": trial.suggest_categorical('distil', [True, False]),
        'embed_type': trial.suggest_categorical('embed_type', ['fixed', 'timeF', 'learned']),
        'freq': trial.suggest_categorical('freq', ['s', 't', 'h', 'd', 'b', 'w', 'm']),
        "use_time2vec": trial.suggest_categorical("use_time2vec", [True, False]),
        "use_fourier_features": trial.suggest_categorical("use_fourier_features", [True, False]),
        "use_sparsemax": trial.suggest_categorical("use_sparsemax", [True, False]),
        "l1_reg": trial.suggest_float("l1_reg", 1e-6, 1e-3, log=True),
        "l2_reg": trial.suggest_float("l2_reg", 1e-6, 1e-3, log=True),
        "use_gnn": trial.suggest_categorical("use_gnn", [True, False]),  # Usar GNN
        "use_transformer": trial.suggest_categorical("use_transformer", [True, False])  # Usar Transformer
    }
    if model_params["use_time2vec"]:
        model_params["time2vec_dim"] = trial.suggest_categorical("time2vec_dim", [8, 16, 32, 64])
        model_params['time2vec_activation'] = trial.suggest_categorical('time2vec_activation', ['sin', 'cos', 'relu'])
    if model_params["use_fourier_features"]:
        model_params["num_fourier_features"] = trial.suggest_int("num_fourier_features", 5, 50)
     #Parametros de la GNN (si se usa)
    if model_params["use_gnn"]:
        gnn_config = {
            'n_hidden': [trial.suggest_categorical("gnn_hidden", [16, 32, 64]) for _ in
                         range(trial.suggest_int("gnn_layers", 1, 3))],  # Lista de dimensiones
            'n_classes': model_params['d_model'],  # El GNN layer retorna un embedding de dimension d_model
            'n_layers': trial.suggest_int("gnn_layers", 1, 3),
            'dropout_rate': model_params['dropout_rate'],  # Mismo dropout
            'gnn_type': trial.suggest_categorical("gnn_type", ["GCN", "GAT", "GraphSAGE"]),
            'use_batchnorm': trial.suggest_categorical('gnn_use_batchnorm', [True, False]),
            'activation': trial.suggest_categorical('gnn_activation', ['relu', 'elu', 'sigmoid']),
            'l2_reg': model_params['l2_reg']  # Misma regularizacion
        }
        #Si se usa la GNN, se debe setear la dimension del embedding
        model_params['gnn_embedding_dim'] = gnn_config['n_hidden'][-1] #Ultima capa

    #Parametros del transformer
    if model_params['use_transformer']:
        model_params['transformer_embedding_dim'] = 768 #Por defecto

    # --- Crear el Modelo ---
    #Instanciar GNN si se usa
    if model_params['use_gnn']:
        gnn = GNN(**gnn_config)
    model = Informer(config=InformerConfig(**{**base_model_params, **model_params})) #Pasamos parametros

    # --- Cargar y Preparar Datos ---
    train_data, val_data, _ = load_and_preprocess_data(config["data_dir"], config=config,
                                                        use_gnn=model_params['use_gnn'],
                                                        use_transformer=model_params['use_transformer'])

    # --- Optimizador y Función de Pérdida ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"])
    criterion = tf.keras.losses.MeanSquaredError()  # Usar MSE por defecto (Informer)

    # --- Entrenamiento (Medir Tiempo) ---
    start_time = time.time()
    for epoch in range(training_params["epochs"]):
        train_loss = train_one_epoch(model, train_data, optimizer, criterion, training_params["batch_size"])
        val_loss = evaluate_epoch(model, val_data, criterion, training_params["batch_size"])

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()  # Terminar si el pruner lo indica
    end_time = time.time()
    training_time = end_time - start_time

    # --- Inferencia (Medir Tiempo) ---
    inference_data = val_data.batch(training_params["batch_size"]).take(1)
    start_time = time.time()
    for inputs, _ in inference_data:
        model(inputs, training=False)
    end_time = time.time()
    inference_time = (end_time - start_time) / training_params["batch_size"]

    return val_loss, inference_time

def optimize_informer(n_trials: int = 100, data_dir: str = "data/processed",
                     config_path="config/informer/informer_base.yaml") -> optuna.Study:
    """Optimiza los hiperparámetros del Informer (multi-objetivo)."""

    # --- Crear un Estudio de Optuna ---
    study = optuna.create_study(directions=["minimize", "minimize"],  # Minimizar pérdida y tiempo
                              pruner=optuna.pruners.HyperbandPruner(),  # Pruner avanzado
                              sampler=optuna.samplers.TPESampler(multivariate=True),  # Considerar interacciones
                              study_name="Informer_Optimization")  # Dar un nombre

    # --- Ejecutar la Optimización ---
    study.optimize(lambda trial: objective(trial, config_path), n_trials=n_trials,
                   catch=(ValueError, RuntimeError))  # Capturar excepciones

    print("Mejores Trials (Frente de Pareto):")
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Valores (Pérdida, Tiempo): {trial.values}")
        print(f"    Parámetros: {trial.params}")

    return study