import optuna
import tensorflow as tf
from models.tft.base.model import TFT  # Importar TFT base
#Si se usan variantes:
from models.tft.variants.tft_indrnn.model import TFTIndRNN
from models.tft_gnn.model import TFT_GNN
from models.tft.base.training import train_one_epoch, evaluate_epoch, quantile_loss, evidential_loss, mdn_loss
from pipelines.preprocessing.preprocess_data import load_and_preprocess_data
from core.utils.helpers import load_config
from models.tft.base.config import TFTConfig
import time
import numpy as np
from typing import Tuple

def objective(trial: optuna.trial.Trial, config_path: str = "config/tft_config.yaml") -> Tuple[float, float]:
    """Función objetivo para Optuna (multi-objetivo)."""

    # --- Cargar Configuración Base ---
    config = load_config(config_path)  # Usamos el path
    base_model_params = config['model_params']
    training_params = config['training_params']

    # --- Hiperparámetros (Espacio de Búsqueda Ampliado) ---
    # (¡Asegúrate de que *todos* los hiperparámetros relevantes estén aquí!)
    model_params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512, 1024]),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 5),
        "attention_heads": trial.suggest_categorical("attention_heads", [1, 2, 4, 8, 16]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.7),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "use_positional_encoding": trial.suggest_categorical("use_positional_encoding", [True, False]),
        "use_dropconnect": trial.suggest_categorical("use_dropconnect", [True, False]),
        "use_scheduled_drop_path": trial.suggest_categorical("use_scheduled_drop_path", [True, False]),
        "drop_path_rate": trial.suggest_float("drop_path_rate", 0.0, 0.5),
        "kernel_initializer": trial.suggest_categorical("kernel_initializer",
                                                        ["glorot_uniform", "he_normal", "lecun_normal", "glorot_normal",
                                                         "he_uniform"]),
        "use_glu_in_grn": trial.suggest_categorical("use_glu_in_grn", [True, False]),
        "use_layer_norm_in_grn": trial.suggest_categorical("use_layer_norm_in_grn", [True, False]),
        "use_multi_query_attention": trial.suggest_categorical("use_multi_query_attention", [True, False]),
        "use_indrnn": trial.suggest_categorical("use_indrnn", [True, False]),
        "use_logsparse_attention": trial.suggest_categorical("use_logsparse_attention", [True, False]),
        "sparsity_factor": trial.suggest_categorical("sparsity_factor", [2, 4, 8, 16]),
        "use_evidential_regression": trial.suggest_categorical("use_evidential_regression", [True, False]),
        "use_mdn": trial.suggest_categorical("use_mdn", [True, False]),
        "num_mixtures": trial.suggest_int("num_mixtures", 2, 10),
        "use_time2vec": trial.suggest_categorical("use_time2vec", [True, False]),
        "use_fourier_features": trial.suggest_categorical("use_fourier_features", [True, False]),
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
            'n_classes': model_params['hidden_size'],  # El GNN layer retorna un embedding de dimension hidden_size
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

    # --- Restricciones (Ejemplo) ---
    if model_params["use_indrnn"] and model_params["use_dropconnect"]:
        trial.set_user_attr("invalid_combination", True)  # Marcar como inválido
        return np.inf, np.inf  # Retornar infinito (o un valor muy alto)

    # --- Crear el Modelo ---
    # (Usar una función factoría, o seleccionar la clase correcta)
    if model_params['use_gnn']:
        model = TFT_GNN(TFTConfig(**{**base_model_params, **model_params}), gnn_config)  # Pasar parametros base y del trial
    elif model_params["use_indrnn"]:
        model = TFTIndRNN(config=TFTConfig(**{**base_model_params, **model_params}))  # Pasar parametros base y del trial
    # elif ... (otras variantes del TFT) ...
    else:
        model = TFT(config=TFTConfig(**{**base_model_params, **model_params}))  # Pasar parametros base y del trial

    # --- Cargar y Preparar Datos ---
    train_data, val_data, _ = load_and_preprocess_data(config["data_dir"], config=config,
                                                        use_gnn=model_params['use_gnn'],
                                                        use_transformer=model_params['use_transformer'])

    # --- Optimizador y Función de Pérdida ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"])

    if model_params["use_evidential_regression"]:
        criterion = evidential_loss
    elif model_params["use_mdn"]:
        criterion = mdn_loss
    elif base_model_params["num_quantiles"] is not None and base_model_params["num_quantiles"] > 0:
        criterion = lambda y_true, y_pred: quantile_loss(y_true, y_pred, [0.1, 0.5, 0.9])  # Ajusta los cuantiles
    else:
        criterion = tf.keras.losses.MeanSquaredError()

    # --- Entrenamiento (Medir Tiempo) ---
    start_time = time.time()
    for epoch in range(training_params["epochs"]):
        train_loss = train_one_epoch(model, train_data, optimizer, criterion, training_params["batch_size"],
                                      use_adversarial_training=training_params['use_adversarial_training'],
                                      epsilon=training_params['epsilon'])  # Pasar adversarial training
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

def optimize_tft(n_trials: int = 100, data_dir: str = "data/processed",
                 config_path="config/tft_config.yaml") -> optuna.Study:
    """Optimiza los hiperparámetros del TFT (multi-objetivo)."""

    # --- Crear un Estudio de Optuna ---
    study = optuna.create_study(directions=["minimize", "minimize"],  # Minimizar pérdida y tiempo
                              pruner=optuna.pruners.HyperbandPruner(),  # Pruner avanzado
                              sampler=optuna.samplers.TPESampler(multivariate=True),  # Considerar interacciones
                              study_name="TFT_Optimization")  # Dar un nombre

    # --- Ejecutar la Optimización ---
    #Pasamos la configuración
    study.optimize(lambda trial: objective(trial, config_path), n_trials=n_trials,
                   catch=(ValueError, RuntimeError))  # Capturar excepciones

    print("Mejores Trials (Frente de Pareto):")
    for trial in study.best_trials:
        print(f"  Trial {trial.number}:")
        print(f"    Valores (Pérdida, Tiempo): {trial.values}")
        print(f"    Parámetros: {trial.params}")

    return study